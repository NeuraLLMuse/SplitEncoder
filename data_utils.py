import torch
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
import numpy as np
from music21 import converter, note, chord, harmony, meter, stream
import torch.nn.functional as F
from tqdm import tqdm
import pickle

class CSGridMLMDataset(Dataset):
    def __init__(
        self,
        root_dir,
        tokenizer,
        # fixed_length=512,
        frontloading=True,
        refrontload=False,
        name_suffix='MLMH'
    ):
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml') or \
                    file.endswith('.mid') or file.endswith('.midi'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.tokenizer = tokenizer
        # self.fixed_length = fixed_length
        self.frontloading = frontloading
        if self.frontloading:
            # check if file exists and load it
            root_dir = root_dir[:-1] if root_dir[-1] == '/' else root_dir
            frontloaded_file = root_dir + '_' + name_suffix + '.pickle'
            if refrontload or not os.path.isfile(frontloaded_file):
                print('Frontloading data.')
                self.encoded = []
                for data_file in tqdm(self.data_files):
                    try:
                        self.encoded.append( self.tokenizer.encode( data_file ) )
                    except Exception as e: 
                        print('Problem in:', data_file)
                        print(e)
                if frontloaded_file is not None:
                    with open(frontloaded_file, 'wb') as f:
                        pickle.dump(self.encoded, f)
            else:
                print('Loading data file.')
                with open(frontloaded_file, 'rb') as f:
                    self.encoded = pickle.load(f)
    # end init

    def __len__(self):
        if self.frontloading:
            return len(self.encoded)
        else:
            return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        if self.frontloading:
            encoded = self.encoded[idx]
        else:
            data_file = self.data_files[idx]
            encoded = self.tokenizer.encode( data_file )
        return {
            'harmony_ids': encoded['harmony_ids'],
            'attention_mask': encoded['attention_mask'],
            'pianoroll': encoded['pianoroll'],
            'time_signature': encoded['time_signature'],
            'h_density_complexity': encoded['h_density_complexity']
        }
    # end getitem
# end class dataset

def CSGridMLM_collate_fn(batch):
    """
    batch: list of dataset items, each one like:
        {
            'harmony_ids': List[int],
            'attention_mask': List[int],
            'time_sig': List[int],
            'pianoroll': np.ndarray of shape (140, fixed_length)
        }
    """
    harmony_ids = [torch.tensor(item['harmony_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    time_signature = [torch.tensor(item['time_signature'], dtype=torch.float) for item in batch]
    h_density_complexity = [torch.tensor(item['h_density_complexity'], dtype=torch.float) for item in batch]
    pianorolls = [torch.tensor(item['pianoroll'], dtype=torch.float) for item in batch]

    return {
        'harmony_ids': torch.stack(harmony_ids),  # shape: (B, L)
        'attention_mask': torch.stack(attention_mask),  # shape: (B, L)
        'time_signature': torch.stack(time_signature),  # shape: (B, whatever dim)
        'h_density_complexity': torch.stack(h_density_complexity),  # shape: (B, whatever dim)
        'pianoroll': torch.stack(pianorolls),  # shape: (B, 140, T)
    }
# end CSGridMLM_collate_fn


def latent_MH_collate_fn(batch):
    """
    batch: list of dataset items, each one like:
        {
            'harmony_ids': List[int],
            'attention_mask': List[int],
            'time_sig': List[int],
            'pianoroll': np.ndarray of shape (140, fixed_length)
        }
    """
    harmony_ids = [torch.tensor(item['harmony_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    pianorolls = [torch.tensor(item['pianoroll'], dtype=torch.float) for item in batch]
    latents = [torch.tensor(item['latent'], dtype=torch.float) for item in batch]

    return {
        'harmony_ids': torch.stack(harmony_ids),  # shape: (B, L)
        'attention_mask': torch.stack(attention_mask),  # shape: (B, L)
        'pianoroll': torch.stack(pianorolls),  # shape: (B, 140, T)
        'latent': torch.stack(latents),  # shape: (B, latent_dim)
    }
# end latent_MH_collate_fn

# ============================================================
# GRAPH DATASET
# ============================================================

class HarmonicGraphDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        max_segment_bars=2,
        include_melody=False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_segment_bars = max_segment_bars
        self.include_melody = include_melody
    # end init

    def __len__(self):
        return len(self.data)
    # end len

    def __getitem__(self, idx):
        d = self.data[idx]
        while len(d['segments']) <= 0:
            idx = np.random.randint(len(self.data))
            d = self.data[idx]
        segment_idx = np.random.randint(len(d['segments']))
        seg = d['segments'][segment_idx]

        return {

            'piece_idx': idx,
            'segment_idx':idx,

            'bar_start': seg['bar_start'],
            'bar_end': seg['bar_start'],

            'mask_token_positions': seg['mask_token_positions'],
            'pianoroll': seg['pianoroll'],

            'real_harmony_ids': seg['real_segment']['real_harmony_ids'],
            'recomposed_harmony_ids': seg['recomposed_segment']['recomposed_harmony_ids'],

            'real_graph': seg['real_segment']['real_graph'],
            'recomposed_graph': seg['recomposed_segment']['recomposed_graph']
        }
    # end getitem
# end class HarmonicGraphDataset

# ============================================================
# GRAPH COLLATE FN
# ============================================================

def harmonic_graph_collate_fn(batch):

    pianorolls = torch.stack([
        torch.tensor(item['pianoroll'], dtype=torch.float)
        for item in batch
    ])

    real_harmony_ids = torch.stack([
        torch.tensor(item['real_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    recomposed_harmony_ids = torch.stack([
        torch.tensor(item['recomposed_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    real_graphs = Batch.from_data_list([
        item['real_graph']
        for item in batch
    ])

    recomposed_graphs = Batch.from_data_list([
        item['recomposed_graph']
        for item in batch
    ])

    mask_token_positions = torch.stack([
        torch.tensor(item['mask_token_positions'], dtype=torch.bool)
        for item in batch
    ])

    return {

        'pianoroll': pianorolls,

        'real_harmony_ids': real_harmony_ids,
        'recomposed_harmony_ids': recomposed_harmony_ids,

        'real_graph': real_graphs,
        'recomposed_graph': recomposed_graphs,

        'mask_token_positions': mask_token_positions
    }
# end harmonic_graph_collate_fn


# ============================================================
# BiLSTM DATASET
# ============================================================

class HarmonicBiLSTMDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        max_segment_bars=2,
        include_melody=False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_segment_bars = max_segment_bars
        self.include_melody = include_melody
    # end init

    def __len__(self):
        return len(self.data)
    # end len

    def __getitem__(self, idx):
        d = self.data[idx]
        while len(d['segments']) <= 0:
            idx = np.random.randint(len(self.data))
            d = self.data[idx]
        segment_idx = np.random.randint(len(d['segments']))
        seg = d['segments'][segment_idx]

        return {

            'piece_idx': idx,
            'segment_idx':idx,

            'bar_start': seg['bar_start'],
            'bar_end': seg['bar_start'],

            'mask_token_positions': seg['mask_token_positions'],
            'pianoroll': seg['pianoroll'],

            'real_harmony_ids': seg['real_segment']['real_harmony_ids'],
            'recomposed_harmony_ids': seg['recomposed_segment']['recomposed_harmony_ids'],

            'real_bilstm': seg['real_segment']['real_bilstm'],
            'recomposed_bilstm': seg['recomposed_segment']['recomposed_bilstm']
        }
    # end getitem
# end class HarmonicBiLSTMDataset

# ============================================================
# BiLSTM COLLATE FN
# ============================================================

def harmonic_bilstm_collate_fn(batch):

    pianorolls = torch.stack([
        torch.tensor(item['pianoroll'], dtype=torch.float)
        for item in batch
    ])

    real_harmony_ids = torch.stack([
        torch.tensor(item['real_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    recomposed_harmony_ids = torch.stack([
        torch.tensor(item['recomposed_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    real_bilstm_list = [
        torch.as_tensor(
            item['real_bilstm'],
            dtype=torch.float32
        )
        for item in batch
    ]
    real_lengths = torch.tensor(
        [x.shape[0] for x in real_bilstm_list],
        dtype=torch.long
    )
    real_bilstm = pad_sequence(
        real_bilstm_list,
        batch_first=True,
        padding_value=0.0
    )

    recomposed_bilstm_list = [
        torch.as_tensor(
            item['recomposed_bilstm'],
            dtype=torch.float32
        )
        for item in batch
    ]
    recomposed_lengths = torch.tensor(
        [x.shape[0] for x in recomposed_bilstm_list],
        dtype=torch.long
    )
    recomposed_bilstm = pad_sequence(
        recomposed_bilstm_list,
        batch_first=True,
        padding_value=0.0
    )

    mask_token_positions = torch.stack([
        torch.tensor(item['mask_token_positions'], dtype=torch.bool)
        for item in batch
    ])

    return {

        'pianoroll': pianorolls,
        
        'real_harmony_ids': real_harmony_ids,
        'recomposed_harmony_ids': recomposed_harmony_ids,

        'real_bilstm': real_bilstm,
        'real_lengths': real_lengths,

        'recomposed_bilstm': recomposed_bilstm,
        'recomposed_lengths': recomposed_lengths,

        'mask_token_positions': mask_token_positions
    }
# end harmonic_bilstm_collate_fn

# ============================================================
# Token BiLSTM DATASET
# ============================================================

class TokenBiLSTMDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        max_segment_bars=2,
        include_melody=False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_segment_bars = max_segment_bars
        self.include_melody = include_melody
    # end init

    def __len__(self):
        return len(self.data)
    # end len

    def __getitem__(self, idx):
        d = self.data[idx]
        while len(d['segments']) <= 0:
            idx = np.random.randint(len(self.data))
            d = self.data[idx]
        segment_idx = np.random.randint(len(d['segments']))
        seg = d['segments'][segment_idx]
        return {

            'piece_idx': idx,
            'segment_idx':idx,

            'bar_start': seg['bar_start'],
            'bar_end': seg['bar_start'],

            'mask_token_positions': seg['mask_token_positions'],
            'pianoroll': seg['pianoroll'],

            'real_harmony_ids': seg['real_segment']['real_harmony_ids'],
            'recomposed_harmony_ids': seg['recomposed_segment']['recomposed_harmony_ids'],

            'real_ids_segment': seg['real_segment']['real_ids_segment'],
            'recomposed_ids_segment': seg['recomposed_segment']['recomposed_ids_segment'],
        }
    # end getitem
# end class TokenBiLSTMDataset

# ============================================================
# Token COLLATE FN
# ============================================================

def token_bilstm_collate_fn(batch):

    pianorolls = torch.stack([
        torch.tensor(item['pianoroll'], dtype=torch.float)
        for item in batch
    ])

    real_harmony_ids = torch.stack([
        torch.tensor(item['real_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    recomposed_harmony_ids = torch.stack([
        torch.tensor(item['recomposed_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    real_bilstm_list = [
        torch.as_tensor(
            item['real_ids_segment'],
            dtype=torch.long
        )
        for item in batch
    ]
    real_lengths = torch.tensor(
        [x.shape[0] for x in real_bilstm_list],
        dtype=torch.long
    )
    real_bilstm = pad_sequence(
        real_bilstm_list,
        batch_first=True,
        padding_value=0
    )

    recomposed_bilstm_list = [
        torch.as_tensor(
            item['recomposed_ids_segment'],
            dtype=torch.long
        )
        for item in batch
    ]
    recomposed_lengths = torch.tensor(
        [x.shape[0] for x in recomposed_bilstm_list],
        dtype=torch.long
    )
    recomposed_bilstm = pad_sequence(
        recomposed_bilstm_list,
        batch_first=True,
        padding_value=0
    )

    mask_token_positions = torch.stack([
        torch.tensor(item['mask_token_positions'], dtype=torch.bool)
        for item in batch
    ])

    return {

        'pianoroll': pianorolls,

        'real_harmony_ids': real_harmony_ids.squeeze(1),
        'recomposed_harmony_ids': recomposed_harmony_ids.squeeze(1),

        'real_bilstm': real_bilstm,
        'real_lengths': real_lengths,

        'recomposed_bilstm': recomposed_bilstm,
        'recomposed_lengths': recomposed_lengths,

        'mask_token_positions': mask_token_positions
    }
# end token_bilstm_collate_fn