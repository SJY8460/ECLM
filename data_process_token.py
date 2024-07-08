import argparse
import os
import json


def read_file(file_path, is_train=False):
    """Read data file of given path.

    :param file_path: path of data file.
    :param is_train: flag to indicate if it's a training file.
    :return: list of sentence, list of slot and list of intent.
    """
    texts, slots, intents, token_intents = [], [], [], []
    text, slot, token_intent = [], [], []

    with open(file_path, 'r', encoding="utf8") as fr:
        for line in fr.readlines():
            items = line.strip().split()

            if len(items) == 1:
                texts.append(' '.join(text))
                slots.append(slot)
                if is_train:
                    token_intents.append(token_intent)
                if "/" not in items[0]:
                    intents.append(items[0])
                else:
                    new = items[0].split("/")
                    intents.append(new[1])

                # clear buffer lists.
                text, slot, token_intent = [], [], []

            elif len(items) >= 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())
                if is_train:
                    token_intent.append(items[2].strip())

    if is_train:
        return texts, slots, intents, token_intents
    else:
        return texts, slots, intents
    
# def format_data(texts, slots, intents, token_intents=None):
#     formatted_data = []

#     for text, slot, intent, token_intent in zip(texts, slots, intents, token_intents if token_intents else [None] * len(texts)):
#         words = text.split()
#         slot_dict = {}
#         sub_utterance_dict = {}
#         current_slot = None
#         current_value = []
#         current_sub = []
#         current_intent = None

#         # 处理slots，转换为字典格式
#         for word, slot_type in zip(words, slot):
#             if slot_type.startswith("B-"):
#                 if current_slot and current_value:
#                     slot_dict[current_slot] = ' '.join(current_value)
#                 current_slot = slot_type[2:]
#                 current_value = [word]
#             elif slot_type.startswith("I-") and current_slot == slot_type[2:]:
#                 current_value.append(word)
#             else:
#                 if current_slot and current_value:
#                     slot_dict[current_slot] = ' '.join(current_value)
#                 current_slot = None
#                 current_value = []

#         if current_slot and current_value:
#             slot_dict[current_slot] = ' '.join(current_value)

#         # 处理token_intents，生成子句字典
#         if token_intent:
#             for word, ti in zip(words, token_intent):
#                 if ti != "SEP":
#                     current_sub.append(word)
#                     current_intent = ti
#                 else:
#                     if current_sub and current_intent:
#                         sub_utterance_dict[current_intent] = ' '.join(current_sub)
#                     current_sub = []
#                     current_intent = None

#             if current_sub and current_intent:
#                 sub_utterance_dict[current_intent] = ' '.join(current_sub)
#         else:
#             sub_utterance_dict = None

#         formatted_example = {
#             'utterance': text,
#             'sub_utterance': sub_utterance_dict,
#             'intent(s)': intent,
#             'slots': ' '.join(slot),
#             'entity_slots': slot_dict
#         }
#         formatted_data.append(formatted_example)

#     return formatted_data

def format_data(texts, slots, intents, token_intents=None):
    formatted_data = []

    for text, slot, intent, token_intent in zip(texts, slots, intents, token_intents if token_intents else [None] * len(texts)):
        words = text.split()
        slot_dict = {}
        sub_utterance_dict = {}
        current_slot = None
        current_value = []
        current_sub = []
        current_intent = None

        # 处理slots，转换为字典格式
        for word, slot_type in zip(words, slot):
            if slot_type.startswith("B-"):
                if current_slot:
                    # 以列表形式存储所有实体值
                    if current_slot in slot_dict:
                        slot_dict[current_slot].append(' '.join(current_value))
                    else:
                        slot_dict[current_slot] = [' '.join(current_value)]
                current_slot = slot_type[2:]
                current_value = [word]
            elif slot_type.startswith("I-") and current_slot == slot_type[2:]:
                current_value.append(word)
            else:
                if current_slot:
                    if current_slot in slot_dict:
                        slot_dict[current_slot].append(' '.join(current_value))
                    else:
                        slot_dict[current_slot] = [' '.join(current_value)]
                current_slot = None
                current_value = []

        if current_slot:
            if current_slot in slot_dict:
                slot_dict[current_slot].append(' '.join(current_value))
            else:
                slot_dict[current_slot] = [' '.join(current_value)]

        
         # 处理token_intents，生成子句字典
        if token_intent:
            for word, ti in zip(words, token_intent):
                if ti != "SEP":
                    current_sub.append(word)
                    current_intent = ti
                else:
                    if current_sub and current_intent:
                        sub_utterance_dict[current_intent] = ' '.join(current_sub)
                    current_sub = []
                    current_intent = None

            if current_sub and current_intent:
                sub_utterance_dict[current_intent] = ' '.join(current_sub)
        else:
            sub_utterance_dict = None

        formatted_example = {
            'utterance': text,
            'sub_utterance': sub_utterance_dict,
            'intent(s)': intent,
            'slots': ' '.join(slot),
            'entity_slots': slot_dict
        }
        formatted_data.append(formatted_example)

    return formatted_data
def process_dataset(dataset_name, data_dir, output_dir):
    file_paths = {
        'train': os.path.join(data_dir, f'{dataset_name}/train.txt'),
        'dev': os.path.join(data_dir, f'{dataset_name}/dev.txt'),
        'test': os.path.join(data_dir, f'{dataset_name}/test.txt')
    }

    processed_data = {}
    for split in ['train', 'dev', 'test']:
        is_train = split == 'train'
        if is_train:
            texts, slots, intents, token_intents = read_file(file_paths[split], is_train)
            processed_data[split] = format_data(texts, slots, intents, token_intents if is_train else None)
        else:
            texts, slots, intents = read_file(file_paths[split], is_train)
            processed_data[split] = format_data(texts, slots, intents)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split, data in processed_data.items():
        output_file = os.path.join(output_dir, f'{dataset_name}/{split}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ATIS Dataset")
    parser.add_argument("--dataset_name", type=str, default="MixATIS_clean", help="Name of the dataset to process")
    parser.add_argument("--data_dir", type=str, default="/home/shangjian/code/Research/SLU/Uni-MIS/data/", help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default='/home/shangjian/code/Research/Multimodal_LLM/SLM/data/', help="Directory to save processed data")
    
    args = parser.parse_args()
    process_dataset(args.dataset_name, args.data_dir, args.output_dir)
