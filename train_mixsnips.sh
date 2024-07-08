
python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub  -bs 4 
python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub  -bs 4 
python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub  -bs 4 

python infer.py -ifb 32 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/snips/Mistral-7B-Instruct-v0.1_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
# # python infer.py -ifb 32 -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" -pp "./save/model/snips/Baichuan2-7B-Chat_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python infer.py -ifb 32 -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" -pp "./save/model/snips/Llama-2-7b-chat-hf_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
# python infer.py -ifb 32 -md "../dataroot/models/THUDM/chatglm3-6b" -pp "./save/model/snips/chatglm3-6b_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python infer.py -ifb 32 -md "../dataroot/models/lmsys/vicuna-7b-v1.5" -pp "./save/model/snips/vicuna-7b-v1.5_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
