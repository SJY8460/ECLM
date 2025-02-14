# The official code of paper ECLM.

## Requirements
Python 3.8+
PyTorch 2.0+
Additional dependencies specified in requirements.txt

## Train
python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --template_type sub -bs 4

python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub  -bs 4 
Additional scripts specified in train.sh

## Infer
python infer.py -ifb 16 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/atis/Mistral-7B-Instruct-v0.1_sub" --template_type sub

python infer.py -ifb 32 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/snips/Mistral-7B-Instruct-v0.1_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
Additional scripts specified in infer.sh
