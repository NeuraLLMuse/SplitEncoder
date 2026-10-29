import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from torch.utils.data import DataLoader, ConcatDataset
from model import SEMelMask
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
from train_utils import train_melharm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

suffix_prefix = 'melharm'

batchsize = 16
device_name = 'cuda:0'
lr = 1e-4
epochs = 300

train_hook = os.getenv('HOOK_CA_TRAIN')
val_hook = os.getenv('HOOK_CA_VAL')

train_gjt = os.getenv('JAZZ_CA_TRAIN')
val_gjt = os.getenv('JAZZ_CA_VAL')

train_nott = os.getenv('NOTT_CA_TRAIN')
val_nott = os.getenv('NOTT_CA_VAL')

train_wiki = os.getenv('WIKI_CA_TRAIN')
val_wiki = os.getenv('WIKI_CA_VAL')

def main():
    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    print('loading hook')
    train_dataset_hook = CSGridMLMDataset(train_hook, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    val_dataset_hook = CSGridMLMDataset(val_hook, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    print('loading gjt')
    train_dataset_gjt = CSGridMLMDataset(train_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    val_dataset_gjt = CSGridMLMDataset(val_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    print('loading nott')
    train_dataset_nott = CSGridMLMDataset(train_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    val_dataset_nott = CSGridMLMDataset(val_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    print('loading wiki')
    train_dataset_wiki = CSGridMLMDataset(train_wiki, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    val_dataset_wiki = CSGridMLMDataset(val_wiki, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

    train_dataset = ConcatDataset([
        train_dataset_hook,
        train_dataset_gjt,
        train_dataset_nott,
        train_dataset_wiki
    ])
    val_dataset = ConcatDataset([
        val_dataset_hook,
        val_dataset_gjt,
        val_dataset_nott,
        val_dataset_wiki
    ])

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=CSGridMLM_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=CSGridMLM_collate_fn)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    loss_fn=CrossEntropyLoss(ignore_index=-100)

    d_model = 512
    model = SEMelMask(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=8,
        num_layers=8,
        grid_length=80,
        pianoroll_dim=tokenizer.pianoroll_dim,
        device=device,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/pretraining_{suffix_prefix}/', exist_ok=True)
    results_path = f'results/pretraining_{suffix_prefix}/pretraining.csv'

    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs(f'saved_models/{suffix_prefix}_pretrained/', exist_ok=True)
    save_dir = f'saved_models/{suffix_prefix}_pretrained/'
    transformer_path = save_dir + 'pretrained.pt'

    train_melharm(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id
    )

# end main

if __name__ == '__main__':
    main()