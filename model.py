import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, Sequential, Linear, ReLU
import math
from copy import deepcopy
from train_utils import full_to_partial_masking

def sinusoidal_positional_encoding(seq_len, d_model, device):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)
# end sinusoidal_positional_encoding

class TransformerEncoderLayerWithAttn(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None  # place to store the weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # same as parent forward, except we intercept attn_weights
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        if not self.training:
            self.last_attn_weights = attn_weights.detach()  # store for later

        # rest of the computation is copied from TransformerEncoderLayer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
# end TransformerEncoderLayerWithAttn

# ============== SplitEncoder Vanilla ==============
class SEVanilla(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            dropout=0.3,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        self.seq_len = grid_length + grid_length
        
        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length, d_model, device
        )
        self.full_pos = torch.cat([self.shared_pos[:, :self.grid_length, :],
                            self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
            self.encoder_layers.append( encoder_layer )
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(
            self,
            melody_grid,
            harmony_tokens=None,
        ):
        """Run the model, optionally steering harmony-side hidden states.

        Args:
            melody_grid: Tensor (B, grid_length, pianoroll_dim).
            harmony_tokens: Tensor (B, grid_length), e.g. mask tokens except bar tokens.
            get_layers_output: If True, return states by layer

        Returns:
            harmony_logits
        """
        B = melody_grid.size(0)
        device = melody_grid.device

        melody_emb = self.melody_proj(melody_grid)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # For style experiments, explicit mask-token IDs are recommended.
            harmony_emb = torch.zeros(
                B, self.grid_length, self.d_model,
                device=device, dtype=melody_emb.dtype
            )

        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)
        full_seq = full_seq + self.full_pos.to(device=device, dtype=melody_emb.dtype)
        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        encoded = full_seq
        harmony_slice = slice(-self.grid_length, None)

        for layer in self.encoder_layers:
            encoded = layer(encoded)

        encoded = self.output_norm(encoded)
        harmony_output = self.output_head(encoded[:, harmony_slice, :])

        return harmony_output
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SEVanilla