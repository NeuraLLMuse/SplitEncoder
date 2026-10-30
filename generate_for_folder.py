from generate_utils import load_SEVanilla, generate_files_with_nucleus
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pickle
from tqdm import tqdm
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-i', '--folder_in', type=str, help='Folder with midi pieces to harmonize.', required=True)
    parser.add_argument('-o', '--folder_out', type=str, help='Folder to place the generated melodic harmonizations.', required=True)
    parser.add_argument('-c', '--checkpoint', type=str, help='Model path. Defaults to vanilla checkpoint.', required=False)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)

    # Parse the arguments
    args = parser.parse_args()
    
    folder_in = args.folder_in
    folder_in = os.getenv(folder_in)
    folder_out = args.folder_out
    os.makedirs(folder_out, exist_ok=True)
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    
    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    # load model ================================================
    print('loading model')

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    model = load_SEVanilla(
        tokenizer,
        device,
        d_model=512,
        checkpoint_path=checkpoint,
    )
    model.eval()
    files = os.listdir(folder_in)
    for file in tqdm(files):
        input_f_path = os.path.join(folder_in, file)
        piece_name = input_f_path.split('/')[-1].split('.')[0]
        gen_out = generate_files_with_nucleus(
            model,
            tokenizer,
            input_f_path=input_f_path,
            mxl_folder_out=None,
            midi_folder_out=folder_out,
            name_suffix=f'{piece_name}',
            use_constraints=False,
            intertwine_bar_info=True,
            normalize_tonality=True,
            temperature=1.0,
            p=0.9,
            unmasking_order='certain',
            create_gen=True,
            create_real=False
        )
# end main

if __name__ == '__main__':
    main()