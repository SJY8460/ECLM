



# atis
# python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"  --template_type sub -lr 5e-5
   
# python train.py -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"  --template_type sub 
  
# python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"  --template_type sub  -lr 5e-5
  
# # python train.py -md "../dataroot/models/THUDM/chatglm3-6b" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"  --template_type sub 
  
# python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5" --train_file "./data/MixATIS_clean/train.json" --val_file "./data/MixATIS_clean/dev.json" --save_dir "./save/model/atis"  --template_type sub  -lr 5e-5



# # # snips
# python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub 

# python train.py -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub 

# python train.py -md "../dataroot/models/THUDM/chatglm3-6b" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub 

# python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"  --template_type sub 

# python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"  --template_type sub 



#13b

# python train.py -md "../dataroot/models/NousResearch/Llama-2-13b-hf" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"  --template_type sub 
# python train.py -md "../dataroot/models/NousResearch/Llama-2-13b-hf"  --template_type sub 


# python train.py -md "../dataroot/models/lmsys/vicuna-13b-v1.5" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips"  --template_type sub 
# python train.py -md "../dataroot/models/lmsys/vicuna-13b-v1.5"  --template_type sub 



# # # chatglm3
# python train.py -md "../dataroot/models/THUDM/chatglm3-6b" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub 
# python train.py -md "../dataroot/models/THUDM/chatglm3-6b"  --template_type sub 

# test
python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1"  --template_type sub -bs 4
python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf"  --template_type sub  -bs 4
python train.py -md "../dataroot/models/lmsys/vicuna-13b-v1.5"  --template_type sub  -bs 4 

python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub  -bs 4 
python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub  -bs 4 
python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5" --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" --template_type sub  -bs 4 




