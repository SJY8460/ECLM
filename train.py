import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from langchain.prompts import PromptTemplate
from utils import format_text,format_text_sub
from Prompt import data_template, test_template,data_template_sub
import torch
import numpy as np
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch


seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs.



def train(model_id, peft_path, train_file, val_file, save_dir, batch_size, max_steps, learning_rate,template_type):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    if 'chatglm' not in model_id.lower() and 'baichuan2' not in model_id.lower():
        tokenizer.pad_token = tokenizer.eos_token
        
    # 加载数据集
    train_dataset = load_dataset('json', data_files= train_file)
    val_dataset = load_dataset('json', data_files= val_file)
    

    if template_type == 'default':
    # 应用 format_text 到数据集
        prompt = PromptTemplate(template=data_template, input_variables=['utterance'  'intent' 'entity_slots'])
        train_dataset = train_dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
        val_dataset = val_dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
    else:
        prompt = PromptTemplate(template=data_template_sub, input_variables=['utterance' 'sub_utterance' 'intent' 'entity_slots'])
        train_dataset = train_dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=True)})
        val_dataset = val_dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=False)})
        
    print("train_dataset_example : \n ", train_dataset['train']['formatted_text'][0])
    print('val_dataset_example : \n ', val_dataset['train']['formatted_text'][0])

    # 设置模型和训练配置
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    if 'chatglm' in model_id.lower():
        # 多卡bug fix
        # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True, device_map={'':torch.cuda.current_device()})
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True, device_map='auto')
        model.hf_device_map['transformer.output_layer'] = model.hf_device_map['transformer.embedding']
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, 
                    trust_remote_code=True, device_map=model.hf_device_map,use_flash_attention_2 = True)
        
    else: 
        # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, 
        #             trust_remote_code=True, device_map='auto',use_flash_attention_2 = False)
        
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                   device_map='auto',use_flash_attention_2 = False)
        
        # config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
        # no_split_modules = model._no_split_modules
        # print(f"no_split_modules: {no_split_modules}", flush=True)
        # map_list = ["10GB","23GB"]
        # device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)
        # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
        #            device_map=device_map,use_flash_attention_2 = True)

        
    training_args = TrainingArguments(
        output_dir = save_dir + "/SFT/{}{}_{}_{}_LR_{}".format(
            model_id.split('/')[-1], 
            template_type, 
            str(batch_size), 
            'BS', 
            str(learning_rate)
        ),
        per_device_train_batch_size=1,
        learning_rate=learning_rate,
        gradient_accumulation_steps=batch_size,
        logging_steps=50,
        # save_steps=200,
        # max_steps=100,
        max_steps=int(len(train_dataset['train'])/ batch_size),
        optim="paged_adamw_8bit",
        fp16=True,
        run_name=f"baseline-{model_id.split('/')[-1]+'_{}'.format(template_type)}",
        remove_unused_columns=False,  #tranformerer version bug
        report_to="none"
    )


      
    if 'baichuan2' in model_id.lower():
        peft_config = LoraConfig(r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules = ["W_pack", "o_proj"])
    else:
        peft_config = LoraConfig(r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # target_modules=['q_proj','v_proj']
        )
    # 训练
    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset["train"],
    #     eval_dataset=val_dataset["train"],
    #     tokenizer=tokenizer,
    #     peft_config=peft_config,
    #     dataset_text_field="formatted_text",
    #     # max_seq_length=512,
    # )
    
    # model_save_dir = os.path.join(save_dir, model_id.split('/')[-1]+'_{}'.format(template_type))
    # if not os.path.exists(model_save_dir):
    #     os.makedirs(model_save_dir)
    
    # trainer.train()
    # trainer.save_model(model_save_dir)
    
    
    #lisa
    from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
    class DynamicLayerActivationCallback(TrainerCallback):
                def __init__(self, n_layers, interval_steps, model):
                    super().__init__()
                    self.n_layers = n_layers
                    self.interval_steps = interval_steps
                    self.model = model

                    # Determine the way to access layers based on the model type
                    class_to_layers_map = {
                        'LlamaForCausalLM': 'model.model.layers',
                        'Qwen2ForCausalLM': 'model.model.layers',
                        'MistralForCausalLM': 'model.model.layers',
                        'MixtralForCausalLM': 'model.model.layers',
                        'GemmaForCausalLM': 'model.model.layers',
                        'GPT2LMHeadModel': 'model.transformer.h',
                    }
                    model_class_name = self.model.__class__.__name__
                    if model_class_name in class_to_layers_map:
                        self.layers_attribute = class_to_layers_map[model_class_name]
                    else:
                        self.layers_attribute = training_args.lisa_layers_attribute
                    self.total_layers = len(eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers

                    self.active_layers_indices = []

                def freeze_all_layers(self):
                    layers = eval('self.' + self.layers_attribute)  # Dynamically execute to get layers
                    for layer in layers:
                        for param in layer.parameters():
                            param.requires_grad = False

                def on_step_begin(self, args, state, control, **kwargs):
                    # Check if it's time to switch active layers, including at step 0
                    if state.global_step % self.interval_steps == 0:
                        self.switch_active_layers()

                def switch_active_layers(self):
                    # First, disable gradients for all layers
                    self.freeze_all_layers()

                    # Randomly select n_layers to activate
                    layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
                    self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
                    print(f"Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)

                    # Enable gradients only for the selected layers
                    for idx in self.active_layers_indices:
                        for param in layers[idx].parameters():
                            param.requires_grad = True

            # Instantiate the callback
    dynamic_layer_activation_callback = DynamicLayerActivationCallback(
        n_layers=1,                     # Number of layers to activate
        interval_steps=20,               # Step interval to update active layers
        model=model
    )
    trainer_callbacks = []
    trainer_callbacks.append(dynamic_layer_activation_callback)

    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=val_dataset["train"],
    tokenizer=tokenizer,
    # peft_config=peft_config,
    dataset_text_field="formatted_text",
    callbacks=trainer_callbacks
    # max_seq_length=512,
)
    
    model_save_dir = os.path.join(save_dir, model_id.split('/')[-1]+'_{}'.format(template_type))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    trainer.train()
    trainer.save_model(model_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Language Model on ATIS Dataset")
    parser.add_argument("--model_id",  '-md', type=str,default="../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1", help="Pretrained model identifier")
    parser.add_argument("--peft_path", type=str, default="./save/model/Mistral-7B-Instruct-v0.1_default", help="Path to PEFT model weights")
    parser.add_argument("--train_file", type=str, default="./data/MixATIS_clean/train.json", help="Path to the training data file")
    parser.add_argument("--val_file", type=str, default="./data/MixATIS_clean/dev.json", help="Path to the validation data file")
    parser.add_argument("--save_dir", type=str, default="./save/model/atis", help="Directory to save the trained model")
    parser.add_argument("--batch_size",'-bs', type=int, default=1, help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of training steps")
    parser.add_argument("--learning_rate",'-lr', type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--template_type", type=str, choices=["default", "sub"], default="default", help="Type of data template to use ('default' or 'sub')")
    
    args = parser.parse_args()

    train(args.model_id, args.peft_path, args.train_file, \
    args.val_file, args.save_dir, args.batch_size, args.max_steps, args.learning_rate, args.template_type)
