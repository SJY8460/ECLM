import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from langchain.prompts import PromptTemplate
from utils import format_text,format_text_sub,format_text_ab,format_text_ab_plus
from Prompt import data_template, data_template_sub,data_template_ab,data_template_ab_plus
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_dataset(dataset, tokenizer, template_type, prompt, is_train=True):
    if template_type=="default":
        return dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
    elif template_type=="ab":
        return dataset.map(lambda x: {"formatted_text": format_text_ab(x, template=prompt,is_train=is_train)})
    elif template_type=="ab_plus":
        return dataset.map(lambda x: {"formatted_text": format_text_ab_plus(x, template=prompt,is_train=is_train)})
    else:
        return dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=is_train)})

def train(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset('json', data_files=args.train_file)
    val_dataset = load_dataset('json', data_files=args.val_file)

    # 根据 data_ratio 参数裁剪数据集
    train_size = int(len(train_dataset['train']) * args.data_ratio)
    val_size = int(len(val_dataset['train']) * args.data_ratio)
    train_dataset = train_dataset['train'].select(range(train_size))
    val_dataset = val_dataset['train'].select(range(val_size))

    if args.template_type == 'default':
        prompt = PromptTemplate(template=data_template, input_variables=['utterance', 'intent', 'entity_slots'])
        train_dataset = format_dataset(train_dataset, tokenizer, args.template_type, prompt)
        val_dataset = format_dataset(val_dataset, tokenizer, args.template_type, prompt)
    elif args.template_type == 'ab':
        prompt = PromptTemplate(template=data_template_ab, input_variables=['utterance', 'sub_utterance', 'intent', 'slots'])
        train_dataset = format_dataset(train_dataset, tokenizer, args.template_type, prompt, is_train=True)
        val_dataset = format_dataset(val_dataset, tokenizer, args.template_type, prompt, is_train=False)
    elif args.template_type == 'ab_plus':
        prompt = PromptTemplate(template=data_template_ab_plus, input_variables=['utterance', 'intent', 'slots'])
        train_dataset = format_dataset(train_dataset, tokenizer, args.template_type, prompt, is_train=True)
        val_dataset = format_dataset(val_dataset, tokenizer, args.template_type, prompt, is_train=False)
    else:
        prompt = PromptTemplate(template=data_template_sub, input_variables=['utterance', 'sub_utterance', 'intent', 'entity_slots'])
        train_dataset = format_dataset(train_dataset, tokenizer, args.template_type, prompt, is_train=True)
        val_dataset = format_dataset(val_dataset, tokenizer, args.template_type, prompt, is_train=False)

    print("Train dataset example:\n", train_dataset['formatted_text'][0])
    print("Validation dataset example:\n", val_dataset['formatted_text'][0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map='auto',
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "SFT", f"{args.model_id.split('/')[-1]}_{args.template_type}_{args.learning_rate}_{args.epochs}_{args.data_ratio}"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_dir=os.path.join(args.save_dir, "logs"),
        logging_steps=10,
        bf16=True,
        run_name=f"baseline-{args.model_id.split('/')[-1]}_{args.template_type}_{args.data_ratio}",
        remove_unused_columns=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="formatted_text",
    )

    trainer.train()

    model_save_dir = os.path.join(args.save_dir, f"{args.model_id.split('/')[-1]}_{args.template_type}_{args.learning_rate}_{args.epochs}_{args.data_ratio}")
    trainer.save_model(model_save_dir)
    print(f"Model saved to {model_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Language Model on ATIS Dataset")
    parser.add_argument("--model_id", '-md', type=str, default="../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1", help="Pretrained model identifier")
    parser.add_argument("--train_file", type=str, default="./data/MixATIS_clean/train.json", help="Path to the training data file")
    parser.add_argument("--val_file", type=str, default="./data/MixATIS_clean/dev.json", help="Path to the validation data file")
    parser.add_argument("--save_dir", type=str, default="./save/model/atis", help="Directory to save the trained model")
    parser.add_argument("--batch_size", '-bs', type=int, default=1, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", '-lr', type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--template_type", type=str, choices=["default", "sub", "ab", "ab_plus"], default="default", help="Type of data template to use")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    parser.add_argument("--data_ratio", type=float, default=1.0, choices=[0.2, 0.4, 0.6, 0.8, 1.0], help="Ratio of data to use for training and validation")

    args = parser.parse_args()
    train(args)
    
# import argparse
# import os
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from trl import SFTTrainer
# from peft import LoraConfig
# from langchain.prompts import PromptTemplate
# from utils import format_text,format_text_sub,format_text_ab,format_text_ab_plus
# from Prompt import data_template, data_template_sub,data_template_ab,data_template_ab_plus
# import torch
# import numpy as np
# import random

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
    
#     # 确保所有 GPU 的随机性一致
#     torch.backends.cudnn.deterministic = True
    
#     # 禁用 cuDNN 的自动优化，确保结果一致
#     torch.backends.cudnn.benchmark = False
    
#     # # 保证算法确定性，尽可能减少随机性
#     # torch.use_deterministic_algorithms(True)


# def format_dataset(dataset, tokenizer, template_type, prompt,is_train=True):
#     if template_type=="default":
#         return dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
#     elif template_type=="ab":
#         return dataset.map(lambda x: {"formatted_text": format_text_ab(x, template=prompt,is_train=is_train)})
#     elif template_type=="ab_plus":
#         return dataset.map(lambda x: {"formatted_text": format_text_ab_plus(x, template=prompt,is_train=is_train)})
#     else:
#         return dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=is_train)})

# def train(args):
#     set_seed(args.seed)

#     tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token

#     train_dataset = load_dataset('json', data_files=args.train_file)
#     val_dataset = load_dataset('json', data_files=args.val_file)

#     if args.template_type == 'default':
#     # 应用 format_text 到数据集
#         prompt = PromptTemplate(template=data_template, input_variables=['utterance'  'intent' 'entity_slots'])
#         train_dataset = format_dataset(train_dataset['train'], tokenizer, args.template_type, prompt)
#         val_dataset = format_dataset(val_dataset['train'], tokenizer, args.template_type, prompt)
#     elif args.template_type == 'ab':
#     # 应用 format_text 到数据集
#         prompt = PromptTemplate(template=data_template_ab, input_variables=['utterance' 'sub_utterance' 'intent' 'slots'])
#         train_dataset = format_dataset(train_dataset['train'], tokenizer, args.template_type, prompt,is_train=True)
#         val_dataset = format_dataset(val_dataset['train'], tokenizer, args.template_type, prompt,is_train=False)
#     elif args.template_type == 'ab_plus':
#     # 应用 format_text 到数据集
#         prompt = PromptTemplate(template=data_template_ab_plus, input_variables=['utterance', 'intent' 'slots'])
#         train_dataset = format_dataset(train_dataset['train'], tokenizer, args.template_type, prompt,is_train=True)
#         val_dataset = format_dataset(val_dataset['train'], tokenizer, args.template_type, prompt,is_train=False)
#     else:
#         prompt = PromptTemplate(template=data_template_sub, input_variables=['utterance' 'sub_utterance' 'intent' 'entity_slots'])
#         train_dataset = format_dataset(train_dataset['train'], tokenizer, args.template_type, prompt,is_train=True)
#         val_dataset = format_dataset(val_dataset['train'], tokenizer, args.template_type, prompt,is_train=False)

#     print("Train dataset example:\n", train_dataset['formatted_text'][0])
#     print("Validation dataset example:\n", val_dataset['formatted_text'][0])

#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_id,
#         device_map='auto',
#         use_flash_attention_2=True,
#         torch_dtype=torch.bfloat16
#     )

#     training_args = TrainingArguments(
#         output_dir=os.path.join(args.save_dir, "SFT", f"{args.model_id.split('/')[-1]}_{args.template_type}_{args.learning_rate}__{args.epochs}"),
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         learning_rate=args.learning_rate,
#         num_train_epochs=args.epochs,
#         # evaluation_strategy="epoch",
#         save_strategy="epoch",
#         logging_dir=os.path.join(args.save_dir, "logs"),
#         logging_steps=10,
#         bf16=True,
#         run_name=f"baseline-{args.model_id.split('/')[-1]}_{args.template_type}",
#         remove_unused_columns=True,
#         report_to="none"
#     )

#     trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         # peft_config=peft_config,
#         dataset_text_field="formatted_text",
#     )

#     trainer.train()

#     model_save_dir = os.path.join(args.save_dir, f"{args.model_id.split('/')[-1]}_{args.template_type}_{args.learning_rate}_{args.epochs}")
#     trainer.save_model(model_save_dir)
#     print(f"Model saved to {model_save_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Training Language Model on ATIS Dataset")
#     parser.add_argument("--model_id" , '-md',type=str, default="../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1", help="Pretrained model identifier")
#     parser.add_argument("--train_file", type=str, default="./data/MixATIS_clean/train.json", help="Path to the training data file")
#     parser.add_argument("--val_file", type=str, default="./data/MixATIS_clean/dev.json", help="Path to the validation data file")
#     parser.add_argument("--save_dir", type=str, default="./save/model/atis", help="Directory to save the trained model")
#     parser.add_argument("--batch_size", '-bs', type=int, default=1, help="Batch size for training and evaluation")
#     parser.add_argument("--learning_rate",'-lr', type=float, default=1e-4, help="Learning rate for training")
#     parser.add_argument("--template_type", type=str, choices=["default", "sub","ab","ab_plus"], default="default", help="Type of data template to use")
#     parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
#     parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")

#     args = parser.parse_args()
#     train(args)