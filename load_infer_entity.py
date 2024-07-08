import ast
from datasets import load_dataset
from utils import parse_generated_text, convert_dict_to_slots, get_multi_acc, computeF1Score, semantic_acc, format_text


def extract_entity_slots(slot_sequence, sentence):
    """
    从标记的slot序列和句子中提取实体槽信息。
    :param slot_sequence: 标记的slot序列，例如 ['B-location', 'O', 'B-person']
    :param sentence: 相应的句子
    :return: entity_slots字典，例如 {'location': ['London'], 'person': ['Alice']}
    """
    entity_slots = {}
    words = sentence.split()
    current_entity = []
    current_slot_type = None

    for word, tag in zip(words, slot_sequence):
        if tag.startswith('B-'):
            # 如果遇到新实体的开始，保存当前实体并开始新的
            if current_entity:
                entity_slots[current_slot_type].append(' '.join(current_entity))
                current_entity = []

            current_slot_type = tag[2:]
            current_entity.append(word)

            if current_slot_type not in entity_slots:
                entity_slots[current_slot_type] = []

        elif tag.startswith('I-') and current_entity:
            # 继续收集当前实体的一部分
            current_entity.append(word)

        else:
            # 非实体部分，保存当前实体（如果有）
            if current_entity:
                entity_slots[current_slot_type].append(' '.join(current_entity))
                current_entity = []
                current_slot_type = None

    # 检查最后一个实体是否需要添加
    if current_entity:
        entity_slots[current_slot_type].append(' '.join(current_entity))

    return entity_slots

def compute_entity_slot_accuracy(pred_slots, true_slots, sentences):
    correct_count = 0
    total_count = 0

    for pred_slot_seq, true_slot_seq, sentence in zip(pred_slots, true_slots, sentences):
        pred_slot_dict = extract_entity_slots(pred_slot_seq, sentence)
        true_slot_dict = extract_entity_slots(true_slot_seq, sentence)  # 使用相同的方法提取真实的实体槽
        total_count += 1
        if pred_slot_dict == true_slot_dict:
            correct_count += 1
       
        # for entity_type, values in true_slot_dict.items():
        #     total_count += len(values)
        #     correct_count += sum(value in pred_slot_dict.get(entity_type, []) for value in values)
    return correct_count / total_count


def load_intents_slots_from_file(file_path, intent_num=None):
    with open(file_path, 'r') as file:
        data = file.read()

    # 分割数据以提取不同部分
    parts = data.split('\n\n')
    pred_intents = ast.literal_eval(parts[0].split('\n', 1)[1])
    true_intents = ast.literal_eval(parts[1].split('\n', 1)[1])
    pred_slots = ast.literal_eval(parts[2].split('\n', 1)[1])
    true_slots = ast.literal_eval(parts[3].split('\n', 1)[1])

    # 如果提供了intent_num，则根据true_intents的长度为intent_num提取对应的意图和槽
    if intent_num is not None:
        filtered_indices = [i for i, intent in enumerate(true_intents) if len(intent) == intent_num]
        pred_intents = [pred_intents[i] for i in filtered_indices]
        true_intents = [true_intents[i] for i in filtered_indices]
        pred_slots = [pred_slots[i] for i in filtered_indices]
        true_slots = [true_slots[i] for i in filtered_indices]

    return pred_intents, true_intents, pred_slots, true_slots


# 使用函数提取数据
file_path = '/home/shangjian/code/Research/Multimodal_LLM/SLM/save/result/MixATIS_clean/Mistral-7B-Instruct-v0.1_sub_checkpoint_1/intents_slots.txt'
pred_intents, true_intents, pred_slots, true_slots = load_intents_slots_from_file(file_path)

# 加载测试数据集
test_dataset = load_dataset('json', data_files='/home/shangjian/code/Research/Multimodal_LLM/SLM/data/MixATIS_clean/test.json')
sentences = [item['utterance'] for item in test_dataset['train']]

# 现在您可以使用这些数据进行测试
intent_acc = get_multi_acc(pred_intents, true_intents)
slot_score = computeF1Score(true_slots, pred_slots)
semantic_accuracy = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
entity_slot_accuracy = compute_entity_slot_accuracy(pred_slots, true_slots, sentences)

# 计算综合语义准确率
combined_semantic_accuracy = intent_acc * entity_slot_accuracy



print(f"Intent Accuracy: {intent_acc}")
print(f"Slot_Score(f1, precision, recall): {slot_score}")
print(f"Semantic Accuracy: {semantic_accuracy}")
print(f"Entity Slot Accuracy: {entity_slot_accuracy}")
print(f"Combined Semantic Accuracy: {combined_semantic_accuracy}")
# # 示例用法
# sentence = "I am traveling to London with Alice"
# slot_sequence = ['O', 'O', 'O', 'O', 'B-location', 'I-location', 'B-person']
# extracted_entity_slots = extract_entity_slots(slot_sequence, sentence)
# print(extracted_entity_slots)
