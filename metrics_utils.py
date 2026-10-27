import torch
import torch.nn.functional as F

def compute_normalized_token_entropy(logits, target_ids, pad_token_id=None):
    """
    Computes Expected Bits per Token (Token Entropy) for a batch.
    
    Args:
        logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (batch_size, seq_len).
        pad_token_id (int, optional): Token ID for padding. If provided, masked out in computation.
        
    Returns:
        entropy_per_token (torch.Tensor): Average entropy per token for each sequence.
        entropy_per_batch (float): Average entropy per token across the batch.
    """
    # Infer vocabulary size from logits shape
    vocab_size = logits.shape[-1]
    # Compute max possible entropy for normalization
    max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32)).item()

    # Compute probabilities with softmax
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Compute log probabilities (base 2)
    log_probs = torch.log2(probs + 1e-9)  # Avoid log(0) errors

    # Compute entropy: H(x) = - sum(P(x) * log2(P(x)))
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)

    # Mask out padding tokens if provided
    if pad_token_id is not None:
        mask = (target_ids != pad_token_id).float()  # 1 for valid tokens, 0 for padding
        entropy = entropy * mask  # Zero out entropy for padding
        entropy_per_token = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # Normalize per valid token
    else:
        entropy_per_token = entropy.mean(dim=-1)  # Average over sequence length

    # Compute overall batch entropy
    entropy_per_batch = entropy_per_token.mean().item()

    return entropy_per_token/max_entropy, entropy_per_batch/max_entropy
# end compute_token_entropy