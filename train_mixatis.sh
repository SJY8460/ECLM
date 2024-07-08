# CUDA_VISIBLE_DEVICES=1 
python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1"  --template_type sub -bs 3 -lr 1e-4
# python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf"  --template_type sub  -bs 4
# python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5"  --template_type sub  -bs 4 -lr 2e-4


python infer.py -ifb 16 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/atis/Mistral-7B-Instruct-v0.1_sub" --template_type sub
# python infer.py -ifb 16 -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" -pp "./save/model/atis/Baichuan2-7B-Chat_sub" --template_type sub
# python infer.py -ifb 16 -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" -pp "./save/model/atis/Llama-2-7b-chat-hf_sub" --template_type sub
# python infer.py -ifb 16 -md "../dataroot/models/THUDM/chatglm3-6b" -pp "./save/model/atis/chatglm3-6b_sub" --template_type sub
# python infer.py -ifb 16 -md "../dataroot/models/lmsys/vicuna-7b-v1.5" -pp "./save/model/atis/vicuna-7b-v1.5_sub" --template_type sub





python train.py -md "../dataroot/models/NousResearch/Llama-2-13b-hf"  --template_type sub -bs 3 -lr 1e-4
# python train.py -md "../dataroot/models/lmsys/vicuna-13b-v1.5"  --template_type sub -bs 2 -lr 1e-4

python infer.py -ifb 16 -md "../dataroot/models/NousResearch/Llama-2-13b-hf" -pp "./save/model/atis/Llama-2-13b-hf_sub" --template_type sub
# python infer.py -ifb 16 -md "../dataroot/models/lmsys/vicuna-13b-v1.5"  -pp "./save/model/atis/vicuna-13b-v1.5_sub" --template_type sub

