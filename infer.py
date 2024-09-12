import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from tqdm import tqdm
from Prompt import test_template, data_template, data_template_sub,data_template_ab,data_template_ab_plus
from utils import parse_generated_text, convert_dict_to_slots,convert_to_slots, get_multi_acc, computeF1Score, semantic_acc, format_text,format_text_ab,format_text_sub,format_text_ab_plus
from langchain.prompts import PromptTemplate

def load_model(model_id, peft_path, device_map='auto', torch_dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16
    )

    # model = PeftModel.from_pretrained(model, peft_path)
    return model, tokenizer


def format_dataset(dataset, tokenizer, template_type, prompt):
    if template_type == "ab":
        return dataset.map(lambda x: {"formatted_text": format_text_ab(x, template=prompt,is_train=False)})
    elif template_type == "sub":
        return dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=False)})
    if template_type == "ab_plus":
        return dataset.map(lambda x: {"formatted_text": format_text_ab_plus(x, template=prompt,is_train=False)})
    else:
        return dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
        

def infer_and_evaluate(test_dataset, test_template, model, tokenizer, generation_config, infer_batch_size):
    all_pred_intents, all_true_intents, all_pred_slots, all_true_slots = [], [], [], []
    texts = []
    for i in range(len(test_dataset)):
        texts.append(test_template.format(utterance=test_dataset[i]["utterance"]))
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), infer_batch_size), desc="Processing"):
            prompts = texts[i:i + infer_batch_size]
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            
            generation_outputs = model.generate(**model_inputs, generation_config=generation_config, max_length=512, return_dict_in_generate=True, 
                output_scores=False, pad_token_id=tokenizer.eos_token_id).sequences.cpu()
            
            for idx, output in enumerate(tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)):

                if args.template_type == 'ab' or args.template_type == 'ab_plus':
                    # 直接处理 slot 标签
                    # print(output)
                    # exit(0)
                    pred_intents, pred_slots, utterance0 = parse_generated_text(output, entity=False)
                    true_intent, true_slots, utterance1 = parse_generated_text(test_dataset[i+idx]['formatted_text'], entity=False)

                    # 将槽位标签转换为 BIO 格式
                    pred_bio_slots = convert_to_slots(pred_slots, test_dataset[i+idx]['utterance'])
                    true_bio_slots = true_slots
                
                else:
                    pred_intents, pred_slots, utterance0 = parse_generated_text(output)
                    true_intent, true_slots, utterance1 = parse_generated_text(test_dataset[i+idx]['formatted_text'])
                    
                    pred_bio_slots = convert_dict_to_slots(pred_slots, test_dataset[i+idx]['utterance'])
                    true_bio_slots = convert_dict_to_slots(true_slots, test_dataset[i+idx]['utterance'])
                    
                    # supervising
                if i % 5 == 0 and idx == infer_batch_size - 1:
                    print("Utterance:")
                    print(utterance0)
                    print(utterance1)
                    print(f"pred_intents: {pred_intents}")
                    print(f"true_intent: {true_intent}")
                    print("Pre_Entity_Slot: ", pred_slots)
                    print("True_Entity_Slot: ", true_slots)
                    print(f"pred_bio_slots: {pred_bio_slots}")
                    print(f"true_bio_slots: {true_bio_slots}")

                all_pred_intents.append(pred_intents)
                all_true_intents.append(true_intent)
                all_pred_slots.append(pred_bio_slots)
                all_true_slots.append(true_bio_slots)

    return all_pred_intents, all_true_intents, all_pred_slots, all_true_slots

def save_results(model_id, checkpoint_num, results, save_dir='./save/result', template_type='default'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    folder_path = os.path.join(save_dir, f'{model_id.split("/")[-1]}_{template_type}_checkpoint_{checkpoint_num}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_path = os.path.join(folder_path, 'results.txt')
    with open(file_path, 'a') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")
    
def save_intents_slots_results(model_id, checkpoint_num, pred_intents, true_intents, pred_slots, true_slots, save_dir='./save/intents_slots', template_type='default'):
    folder_path = os.path.join(save_dir, f'{model_id.split("/")[-1]}_{template_type}_checkpoint_{checkpoint_num}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, 'intents_slots.txt')
    with open(file_path, 'w') as file:
        file.write("Predicted Intents:\n")
        file.write(str(pred_intents) + "\n\n")
        file.write("True Intents:\n")
        file.write(str(true_intents) + "\n\n")
        file.write("Predicted Slots:\n")
        file.write(str(pred_slots) + "\n\n")
        file.write("True Slots:\n")
        file.write(str(true_slots) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and Evaluation Script")
    parser.add_argument("--model_id", '-md', type=str, default='../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1', help="Pretrained model identifier")
    parser.add_argument("--peft_path", '-pp', type=str, default="./save/model/atis/Mistral-7B-Instruct-v0.1_default", help="Path to PEFT model weights")
    parser.add_argument("--data_file", type=str, default="./data/MixATIS_clean/test.json", help="Path to the test data file")
    parser.add_argument("--infer_batch_size", '-ifb', type=int, default=1, help="Inference batch size")
    parser.add_argument("--checkpoint_num", type=int, default=1, help="Checkpoint number for saving results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.75, help="Top p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=40, help="Top k for generation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--template_type", type=str, choices=["default", "sub","ab","ab_plus"], default="default", help="Type of data template to use ('default' or 'sub')")
    args = parser.parse_args()


    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=False,
        # top_p=args.top_p,
        # top_k=args.top_k,
        # num_beams=args.num_beams,
    )

    model, tokenizer = load_model(args.model_id, args.peft_path)
    test_dataset = load_dataset('json', data_files=args.data_file)
    
    if args.template_type == 'default':
        template = data_template
        input_vars = ['utterance', 'intent', 'entity_slots']

    elif args.template_type == 'ab':
        template = data_template_ab
        input_vars = ['utterance', 'sub_utterance','intent', 'slots']

    elif args.template_type == 'ab_plus':
        template = data_template_ab_plus
        input_vars = ['utterance','intent', 'slots']

    else:
        template = data_template_sub
        input_vars = ['utterance', 'sub_utterance', 'intent', 'entity_slots']

    # 根据 template_type 创建 PromptTemplate
    prompt = PromptTemplate(
        template=template,
        input_variables=input_vars
    )

    test_dataset = format_dataset(test_dataset['train'], tokenizer, args.template_type, prompt)
    
    print("test_dataset_example:\n", test_dataset['formatted_text'][0])
    
    pred_intents, true_intents, pred_slots, true_slots = infer_and_evaluate(test_dataset, test_template, model, tokenizer, generation_config, args.infer_batch_size)
    intent_acc = get_multi_acc(pred_intents, true_intents)
    slot_score = computeF1Score(true_slots, pred_slots)
    semantic_accuracy = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    
    print(f"Intent Accuracy: {intent_acc}")
    print(f"Slot_Score(f1, precision, recall): {slot_score}")
    print(f"Semantic Accuracy: {semantic_accuracy}")

    results = {
        "Intent Accuracy": intent_acc,
        "Slot Score (F1, Precision, Recall)": slot_score,
        "Semantic Accuracy": semantic_accuracy
    }
    
    dataset_name = args.data_file.split('/')[-2]
    save_directory = f'./save/result/{dataset_name}/'

    save_results(args.model_id, args.checkpoint_num, results, save_directory, template_type=args.template_type)
    save_intents_slots_results(args.model_id, args.checkpoint_num, pred_intents, true_intents, 
        pred_slots, true_slots, save_directory, template_type=args.template_type)
