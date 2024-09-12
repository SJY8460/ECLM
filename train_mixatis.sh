python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 2e-5 -bs 32 --epochs 1 --template_type sub  --data_ratio 0.2
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.2" -ifb 128
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 2e-5 -bs 32 --epochs 1 --template_type sub  --data_ratio 0.4
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.4" -ifb 128
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 2e-5 -bs 32 --epochs 1 --template_type sub  --data_ratio 0.6
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.6" -ifb 128
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 2e-5 -bs 32 --epochs 1 --template_type sub  --data_ratio 0.8
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.8" -ifb 128




python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct"  --data_ratio 0.2 -lr 2e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.2" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct"  --data_ratio 0.4 -lr 2e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.4" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct"  --data_ratio 0.6 -lr 2e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.6" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct"  --data_ratio 0.8 -lr 2e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1_0.8" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub



python train.py -md "NousResearch/Llama-2-7b-chat-hf"  --data_ratio 1.0 -lr 2e-5 -bs 32 --epochs 1 --template_type sub 
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Llama-2-7b-chat-hf_sub_2e-05_1_1.0" -ifb 12 --template_type sub

python train.py -md "mistralai/Mistral-7B-Instruct-v0.1" -lr 5e-5 -bs 32 --epochs 1 --template_type sub  --data_ratio 1.0
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Mistral-7B-Instruct-v0.1_sub_5e-05_1_1.0" -ifb 128

python train.py -md "NousResearch/Llama-2-7b-chat-hf"  --data_ratio 1.0 -lr 2e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Llama-2-7b-chat-hf_sub_2e-05_1_1.0" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python train.py -md "meta-llama/Meta-Llama-3.1-8B"  --data_ratio 1.0 -lr 2e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B_sub_2e-05_1_1.0" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub
python train.py -md "mistralai/Mistral-7B-Instruct-v0.1"  --data_ratio 1.0 -lr 5e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Mistral-7B-Instruct-v0.1_sub_5e-05_1_1.0" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub





# CUDA_VISIBLE_DEVICES=1 
# python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 5e-6 -bs 32
# python train.py -md "meta-llama/Meta-Llama-3.1-8B" -lr 4e-5 -bs 32 --epochs 2
# python train.py -md "../dataroot/models/NousResearch/Llama-2-7b-chat-hf"  --template_type sub  -bs 4
# python train.py -md "../dataroot/models/lmsys/vicuna-7b-v1.5"  --template_type sub  -bs 4 -lr 2e-4

python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 4e-5 -bs 32 --epochs 1  --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_default_4e-05_1" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" 

python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 3e-5 -bs 32 --epochs 1 
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_default_3e-05_1" -ifb 128

#main
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 1e-5 -bs 32 --epochs 1 --template_type sub  
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_sub_1e-05_1" -ifb 128
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 2e-5 -bs 32 --epochs 1 --template_type sub --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_sub_2e-05_1" -ifb 128 --data_file "./data/MixSNIPS_clean/test.json" --template_type sub


python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 5e-5 -bs 32 --epochs 1 --template_type ab 
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_ab_5e-05_1" -ifb 128 --template_type ab
python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 5e-5 -bs 32 --epochs 1 --template_type ab --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_ab_5e-05_1"  --template_type ab -ifb 128 --data_file "./data/MixSNIPS_clean/test.json"


python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 3e-5 -bs 32 --epochs 1 --template_type ab_plus 
python infer.py  -md "/scratch/avt2gy/slm/save/model/atis/Meta-Llama-3.1-8B-Instruct_ab_plus_3e-05_1" -ifb 128 --template_type ab_plus

python train.py -md "meta-llama/Meta-Llama-3.1-8B-Instruct" -lr 2e-4 -bs 32 --epochs 1 --template_type ab_plus --train_file "./data/MixSNIPS_clean/train.json" --val_file "./data/MixSNIPS_clean/dev.json" --save_dir "./save/model/snips" 
python infer.py  -md "/scratch/avt2gy/slm/save/model/snips/Meta-Llama-3.1-8B-Instruct_ab_plus_0.0002_1_1.0"  --template_type ab_plus -ifb 128 --data_file "./data/MixSNIPS_clean/test.json"

