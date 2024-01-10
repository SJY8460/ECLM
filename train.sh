
# atis
python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"

python train.py -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"

python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"

python train.py -md "../dataroot/models/THUDM/chatglm3-6b" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"

python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"



# snips
python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"

python train.py -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"

python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"

python train.py -md "../dataroot/models/THUDM/chatglm3-6b" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"

python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"

