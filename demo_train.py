import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from langchain import PromptTemplate
from Prompt import data_template, test_template
from datasets import load_dataset
from utils import format_text
# %%


# %%


# 加载训练、验证和测试数据
data_files = {
    "train": "/home/shangjian/code/Research/Multimodal_LLM/SLM/data/MixATIS_clean/train.json",
    "validation": "/home/shangjian/code/Research/Multimodal_LLM/SLM/data/MixATIS_clean/dev.json",
    "test": "/home/shangjian/code/Research/Multimodal_LLM/SLM/data/MixATIS_clean/test.json"
}

# 加载数据集
train_dataset = load_dataset('json', data_files=data_files['train'])
dev_dataset =  load_dataset('json', data_files=data_files['validation'])
test_dataset =  load_dataset('json', data_files=data_files['test'])

# 查看数据集结构
print(train_dataset,dev_dataset,test_dataset)
print("Length of train dataset: ", len(train_dataset['train']))
print("Length of dev dataset: ", len(dev_dataset['train']))
print("Length of test dataset: ", len(test_dataset['train']))

# %%


prompt = PromptTemplate(template=data_template, input_variables=['utterance'  'intent' 'entity_slots'])

# 应用format_text到数据集
train_dataset = train_dataset.map(lambda x: {"formatted_text":  format_text(x, template=prompt)})
dev_dataset = dev_dataset.map(lambda x: {"formatted_text":  format_text(x, template=prompt)})
test_dataset = test_dataset.map(lambda x: {"formatted_text":  format_text(x, template=prompt)})

# 查看处理后的数据集
print("train_dataset_example:" , train_dataset['train']['formatted_text'][0])
    

# %%
model_id = "/home/shangjian/code/Research/Multimodal_LLM/dataroot/models/Mistral/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# %%
qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# %%
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map='auto'
)

# %%
bs= 2
training_args = TrainingArguments(
    output_dir="./save/SFT/{}".format(model_id.split('/')[-1]) , 
    per_device_train_batch_size=bs,
    learning_rate=2e-4,
    logging_steps=50,
    # do_eval = True,
    # evaluation_strategy = 'steps',
    # eval_steps=10,
    save_steps=150,
    # num_train_epochs= 1,
    logging_strategy="steps",
    # max_steps=int(len(train_dataset['train'])/ bs),
    max_steps=200,
    optim="paged_adamw_8bit",
    fp16=True,
    run_name="baseline-{}".format(model_id.split('/')[-1]),
    remove_unused_columns=False
)

# %%
supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=train_dataset["train"],
    eval_dataset=dev_dataset["train"],
    args=training_args,
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="formatted_text",
    max_seq_length=512,
)

# %%
supervised_finetuning_trainer.train()

# %%
save_dir = "/home/shangjian/code/Research/Multimodal_LLM/SLM/save/model/" + model_id.split('/')[-1] 
supervised_finetuning_trainer.save_model(save_dir)




