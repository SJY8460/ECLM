import ast
from utils import parse_generated_text, convert_dict_to_slots, get_multi_acc, computeF1Score, semantic_acc,format_text

def load_intents_slots_from_file(file_path, intent_num=2):
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
file_path = '/home/shangjian/code/Research/Multimodal_LLM/SLM/save/result/MixATIS_clean/Llama-2-7b-chat-hf_sub_checkpoint_1/intents_slots.txt'
pred_intents, true_intents, pred_slots, true_slots = load_intents_slots_from_file(file_path)

# 现在您可以使用这些数据进行测试
intent_acc = get_multi_acc(pred_intents, true_intents)
slot_score = computeF1Score(true_slots, pred_slots)
semantic_accuracy = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)

print(f"Intent Accuracy: {intent_acc}")
print(f"Slot_Score(f1, precision, recall): {slot_score}")
print(f"Semantic Accuracy: {semantic_accuracy}")
