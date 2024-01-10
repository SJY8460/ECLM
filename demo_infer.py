
# %%
import os
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM 
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
    TrainingArguments,
    GenerationConfig
)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from langchain import PromptTemplate
from Prompt import train_template,test_template
from tqdm import tqdm
from utils import parse_generated_text, convert_dict_to_slots, get_multi_acc, semantic_acc,computeF1Score,format_text



# 加载训练、验证和测试数据
data_files = {
    "train": "/home/shangjian/code/Research/Multimodal & LLM/SLM/data/MixATIS_clean/train.json",
    "validation": "/home/shangjian/code/Research/Multimodal & LLM/SLM/data/MixATIS_clean/dev.json",
    "test": "/home/shangjian/code/Research/Multimodal & LLM/SLM/data/MixATIS_clean/test.json"
}

test_dataset =  load_dataset('json', data_files=data_files['test'])

#  默认train_prmpt
prompt = PromptTemplate(template=train_template, input_variables=['utterance' 'intent' 'entity_slots'])

# 应用format_text到数据集
test_dataset = test_dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = '/home/shangjian/code/Research/Multimodal & LLM/dataroot/models/Mistral/Mistral-7B-Instruct-v0.1'
peft_path = "/home/shangjian/code/Research/Multimodal & LLM/SLM/save/model/" + model_id.split('/')[-1] 

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map='auto'
)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1, # beam search
    do_sample=True
)

# loading peft weight
model = PeftModel.from_pretrained(
    model,
    peft_path,
    torch_dtype=torch.float16,
)

model = model.bfloat16()
model.eval()

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

# 生成回复并计算评估指标
all_pred_intents = []
all_true_intents = []
all_pred_slots = []
all_true_slots = []


infer_batch_size = 1
texts = []
for i in range(0,len(test_dataset["train"])):
    texts.append(test_template.format(utterance=test_dataset["train"][i]["utterance"]))
    
with torch.no_grad():
    for i in tqdm(range(0, len(test_dataset["train"]), infer_batch_size), desc="Processing"):
        
        prompts  = texts[i:i + infer_batch_size]
        
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        generation_outputs = model.generate(
            **model_inputs,
            max_length=256,  # 或其他适当的最大长度
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_sequences = generation_outputs.sequences.cpu()
        
        for idx, output in enumerate(tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)):
            generated_text = output
            # 解析生成的文本以获取预测的意图和槽位
            # print(generated_text)
            pred_intents, pred_slots, _ = parse_generated_text(generated_text)
            
            # print("--------------------",pred_intents,pred_slots)
            
            true_intent, true_slots, utterance = parse_generated_text(test_dataset["train"][i+idx]['formatted_text'])
            pred_bio_slots = convert_dict_to_slots(pred_slots, utterance)

            # 将真实的entity_slots转换为BIO格式
            true_bio_slots = convert_dict_to_slots(true_slots,utterance)

            # 添加预测和真实的意图和槽位到列表
            all_pred_intents.append(pred_intents)
            all_true_intents.append(true_intent)
            all_pred_slots.append(pred_bio_slots)
            all_true_slots.append(true_bio_slots)
        
        
# 计算多意图准确率、槽位F1分数和语义准确率
# 使用之前定义的 get_multi_acc, computeF1Score 和 semantic_acc 函数
intent_acc = get_multi_acc(all_pred_intents, all_true_intents)
slot_score = computeF1Score(all_true_slots, all_pred_slots)
semantic_accuracy = semantic_acc(all_pred_slots, all_true_slots, all_pred_intents, all_true_intents)

# 打印评估指标
print(f"Intent Accuracy: {intent_acc}")
print(f"Slot_Score(f1, precision, recall): {slot_score}")
print(f"Semantic Accuracy: {semantic_accuracy}")

    