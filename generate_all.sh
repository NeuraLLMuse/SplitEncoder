#!/bin/bash

# TODO:
# 1) Correct the transformer model so that it isolates the FiLM
# parameters in a ModuleList.
# 2) Add an argument for selecting contrastive vs MSE loss.

# List of Python scripts with their respective arguments
scripts=(
    "generate_for_folder.py -i JAZZ_VAL -o MIDIs/jazz_f2f2f_12 -c saved_models/f2f2f_pretrained_12/pretrained.pt -n 0 -g 1"
    "generate_for_folder.py -i JAZZ_VAL -o MIDIs/jazz_melharm_12 -c saved_models/melharm_pretrained_12/pretrained.pt -n 0 -g 1"

    "generate_for_folder.py -i NOTT_VAL -o MIDIs/nott_f2f2f_12 -c saved_models/f2f2f_pretrained_12/pretrained.pt -g -n 0 1"
    "generate_for_folder.py -i NOTT_VAL -o MIDIs/nott_melharm_12 -c saved_models/melharm_pretrained_12/pretrained.pt -n 0 -g 1"

    "generate_for_folder.py -i JAZZ_CA_VAL -o MIDIs/jazz_f2f2f_CA -c saved_models/f2f2f_pretrained_CA/pretrained.pt -n 1 -g 1"
    "generate_for_folder.py -i JAZZ_CA_VAL -o MIDIs/jazz_melharm_CA -c saved_models/melharm_pretrained_CA/pretrained.pt -n 1 -g 1"

    "generate_for_folder.py -i NOTT_CA_VAL -o MIDIs/nott_f2f2f_CA -c saved_models/f2f2f_pretrained_CA/pretrained.pt -n 1 -g 1"
    "generate_for_folder.py -i NOTT_CA_VAL -o MIDIs/nott_melharm_CA -c saved_models/melharm_pretrained_CA/pretrained.pt -n 1 -g 1"
)

# Name of the conda environment
conda_env="torch"

# Path to global conda
CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"

for script in "${scripts[@]}"; do
    script_name=$(echo "$script" | awk '{print $1}')
    screen_name=$(basename "$script_name" .py)

    screen -dmS "$screen_name" bash -c "
        source \"$CONDA_SH\"
        conda activate \"$conda_env\"
        python $script
        exec bash
    "

    echo "Started screen '$screen_name' for script '$script'."
done