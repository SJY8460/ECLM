#!/bin/bash

# Baichuan2-7B-Chat 模型
for checkpoint_num in $(seq 1000 1000 39000); do
    python infer.py -ifb 64 -md "../dataroot/models/baichuan-inc/Baichuan2-7B-Chat" -pp "./save/model/snips/SFT/Baichuan2-7B-Chat_default/checkpoint-$checkpoint_num" --data_file "./data/MixSNIPS_clean/test.json" --checkpoint_num $checkpoint_num
done

# Mistral-7B-Instruct-v0.1 模型
for checkpoint_num in $(seq 1000 1000 39000); do
    python infer.py -ifb 64 -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" -pp "./save/model/snips/SFT/Mistral-7B-Instruct-v0.1_default/checkpoint-$checkpoint_num" --data_file "./data/MixSNIPS_clean/test.json" --checkpoint_num $checkpoint_num
done

# Llama-2-7b-chat-hf 模型
for checkpoint_num in $(seq 1000 1000 39000); do
    python infer.py -ifb 64 -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf" -pp "./save/model/snips/SFT/Llama-2-7b-chat-hf_default/checkpoint-$checkpoint_num" --data_file "./data/MixSNIPS_clean/test.json" --checkpoint_num $checkpoint_num
done

# chatglm3-6b 模型
for checkpoint_num in $(seq 1000 1000 39000); do
    python infer.py -ifb 64 -md "../dataroot/models/THUDM/chatglm3-6b" -pp "./save/model/snips/SFT/chatglm3-6b_default/checkpoint-$checkpoint_num" --data_file "./data/MixSNIPS_clean/test.json" --checkpoint_num $checkpoint_num
done

# vicuna-7b-v1.5 模型
for checkpoint_num in $(seq 1000 1000 39000); do
    python infer.py -ifb 64 -md "../dataroot/models/lmsys/vicuna-7b-v1.5" -pp "./save/model/snips/SFT/vicuna-7b-v1.5_default/checkpoint-$checkpoint_num" --data_file "./data/MixSNIPS_clean/test.json" --checkpoint_num $checkpoint_num
done
