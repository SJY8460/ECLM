# 对 ATIS 数据集进行推理
python infer.py -ifb 16 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/atis/Mistral-7B-Instruct-v0.1_sub" --template_type sub
# python infer.py -ifb 16 -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" -pp "./save/model/atis/Baichuan2-7B-Chat_sub" --template_type sub
python infer.py -ifb 16 -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" -pp "./save/model/atis/Llama-2-7b-chat-hf_sub" --template_type sub
# python infer.py -ifb 16 -md "../dataroot/models/THUDM/chatglm3-6b" -pp "./save/model/atis/chatglm3-6b_sub" --template_type sub
python infer.py -ifb 16 -md "../dataroot/models/lmsys/vicuna-7b-v1.5" -pp "./save/model/atis/vicuna-7b-v1.5_sub" --template_type sub

# # 对 SNIPS 数据集进行推理
python infer.py -ifb 32 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/snips/Mistral-7B-Instruct-v0.1_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
# # python infer.py -ifb 32 -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" -pp "./save/model/snips/Baichuan2-7B-Chat_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python infer.py -ifb 32 -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" -pp "./save/model/snips/Llama-2-7b-chat-hf_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
# python infer.py -ifb 32 -md "../dataroot/models/THUDM/chatglm3-6b" -pp "./save/model/snips/chatglm3-6b_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python infer.py -ifb 32 -md "../dataroot/models/lmsys/vicuna-7b-v1.5" -pp "./save/model/snips/vicuna-7b-v1.5_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub


# python infer.py -ifb 16 -md "../dataroot/models/NousResearch/Llama-2-13b-hf" -pp "./save/model/snips/Llama-2-13b-hf_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub

# python infer.py -ifb 16 -md "../dataroot/models/lmsys/vicuna-13b-v1.5" -pp "./save/model/snips/vicuna-13b-v1.5_sub" --data_file "./data/MixSNIPS_clean/test.json" --template_type sub

# python infer.py -ifb 16 -md "../dataroot/models/NousResearch/Llama-2-13b-hf" -pp "./save/model/atis/Llama-2-13b-hf_sub"  --template_type sub

# python infer.py -ifb 16 -md "../dataroot/models/lmsys/vicuna-13b-v1.5" -pp "./save/model/atis/vicuna-13b-v1.5_sub"  --template_type sub



