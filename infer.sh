# 对 ATIS 数据集进行推理
python infer.py -ifb 2 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/atis/Mistral-7B-Instruct-v0.1"
python infer.py -ifb 2 -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" -pp "./save/model/atis/Baichuan2-7B-Chat"
python infer.py -ifb 2 -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" -pp "./save/model/atis/Llama-2-7b-chat-hf"
python infer.py -ifb 2 -md "../dataroot/models/THUDM/chatglm3-6b" -pp "./save/model/atis/chatglm3-6b"
python infer.py -ifb 2 -md "../dataroot/models/lmsys/vicuna-7b-v1.5" -pp "./save/model/atis/vicuna-7b-v1.5"

# 对 SNIPS 数据集进行推理
python infer.py -ifb 2 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/snips/Mistral-7B-Instruct-v0.1"
python infer.py -ifb 2 -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" -pp "./save/model/snips/Baichuan2-7B-Chat"
python infer.py -ifb 2 -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" -pp "./save/model/snips/Llama-2-7b-chat-hf"
python infer.py -ifb 2 -md "../dataroot/models/THUDM/chatglm3-6b" -pp "./save/model/snips/chatglm3-6b"
python infer.py -ifb 2 -md "../dataroot/models/lmsys/vicuna-7b-v1.5" -pp "./save/model/snips/vicuna-7b-v1.5"
