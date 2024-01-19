# %%
import re

def format_text(example, template):
    return template.format(utterance=example['utterance'], intent=example['intent(s)'],
        entity_slots={k: v for k, v in example['entity_slots'].items() if v is not None})

def format_text_sub(example,template,is_train=True):
    if is_train:
        return template.format(utterance=example['utterance'], 
                                   sub_utterance={k: v for k, v in example['sub_utterance'].items() if v is not None},
                                   intent=example['intent(s)'],
                                   entity_slots={k: v for k, v in example['entity_slots'].items() if v is not None})
    else:
        return template.format(utterance=example['utterance'], 
                                  intent=example['intent(s)'],
                                  sub_utterance = "None",
                                  entity_slots={k: v for k, v in example['entity_slots'].items() if v is not None})

def parse_generated_text(generated_text):
    # 使用正则表达式匹配意图和实体槽位
    intent_pattern = r"intent: ([\w#]+)"
    # entity_slots_pattern = r"entity_slot: \{([^}]+)\}"
    entity_slots_pattern = r"entity_slot: \{([^}]+)\}"
    utterance_pattern = r"utterance: (.+)"

    # 提取意图
    intent_match = re.search(intent_pattern, generated_text)
    intents = intent_match.group(1).split('#') if intent_match else []

    # 提取实体槽位
    # entity_slots_match = re.search(entity_slots_pattern, generated_text)
    # entity_slots = {}
    # if entity_slots_match:
    #     slots_str = entity_slots_match.group(1)
    #     for slot_str in slots_str.split(', '):
    #         if ':' in slot_str:
    #             key, value = slot_str.split(': ',1)
    #             entity_slots[key.strip("'")] = value.strip("'")
    
    # 提取实体槽位
    entity_slots_match = re.search(entity_slots_pattern, generated_text)
    entity_slots = {}
    if entity_slots_match:
        slots_str = entity_slots_match.group(1)
        # 使用正则表达式匹配每个键值对
        key_value_pairs = re.findall(r"'([^']+?)': \[([^\]]+?)\]", slots_str)
        for key, values_str in key_value_pairs:
            # 将值分割成列表，去除空格和引号
            values = [value.strip(" '") for value in values_str.split(',')]
            entity_slots[key] = values

    # 提取utterance
    utterance_match = re.search(utterance_pattern, generated_text)
    utterance = utterance_match.group(1).strip() if utterance_match else ""

    return intents, entity_slots, utterance


def convert_dict_to_slots_old(entity_slots, sentence):
    words = sentence.split()
    slot_sequence = ['O'] * len(words)  # 初始化槽位序列为全'O'

    for slot_type, slot_value in entity_slots.items():
        if slot_value:
            slot_words = slot_value.split()
            start_index = find_sublist_index(slot_words, words)

            if start_index != -1:
                # 标记B类型槽位
                slot_sequence[start_index] = f"B-{slot_type}"
                # 标记随后的I类型槽位
                for i in range(start_index + 1, start_index + len(slot_words)):
                    slot_sequence[i] = f"I-{slot_type}"

    return slot_sequence


def find_sublist_index_old(sublist, lst):
    for i in range(len(lst) - len(sublist) + 1):
        if sublist == lst[i:i + len(sublist)]:
            return i
    return -1

def find_sublist_index(sublist, lst, start_index=0):
    for i in range(start_index, len(lst) - len(sublist) + 1):
        if sublist == lst[i:i + len(sublist)]:
            return i
    return -1

def convert_dict_to_slots(entity_slots, sentence):
    words = sentence.split()
    slot_sequence = ['O'] * len(words)

    # 将实体按长度排序，优先处理较长的实体
    sorted_entities = sorted([(slot_type, slot_value.split())
                              for slot_type, slot_values in entity_slots.items()
                              for slot_value in slot_values],
                             key=lambda x: len(x[1]), reverse=True)

    for slot_type, slot_words in sorted_entities:
        word_idx = 0
        while word_idx < len(words):
            start_index = find_sublist_index(slot_words, words, word_idx)

            if start_index != -1:
                # 检查当前位置及后续位置是否已被标记
                if all(tag == 'O' for tag in slot_sequence[start_index:start_index + len(slot_words)]):
                    slot_sequence[start_index] = f"B-{slot_type}"
                    for i in range(1, len(slot_words)):
                        if start_index + i < len(slot_sequence):
                            slot_sequence[start_index + i] = f"I-{slot_type}"
                word_idx = start_index + 1  # 适当更新word_idx
            else:
                break

    return slot_sequence


def get_multi_acc(pred_output, golds):
    acc = 0
    total = 0
    for p, c in zip(pred_output, golds):
        # print(p ,'<=>', c , c == p)
        if set(p) == set(c):
            acc += 1
        total += 1
    return acc / total


# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart


def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def computeF1Score(correct_slots, pred_slots):
    correctChunk = {}
    correctChunkCnt = 0.0
    foundCorrect = {}
    foundCorrectCnt = 0.0
    foundPred = {}
    foundPredCnt = 0.0
    correctTags = 0.0
    tokenCount = 0.0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                 __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                 (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1.0
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1.0
                    else:
                        correctChunk[lastCorrectType] = 1.0
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                 __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                 (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
             __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
             (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType,
                              correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1.0
                else:
                    foundCorrect[correctType] = 1.0

            if __startOfChunk(lastPredTag, predTag, lastPredType,
                              predType) == True:
                foundPredCnt += 1.0
                if predType in foundPred:
                    foundPred[predType] += 1.0
                else:
                    foundPred[predType] = 1.0

            if correctTag == predTag and correctType == predType:
                correctTags += 1.0

            tokenCount += 1.0

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1.0
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1.0
            else:
                correctChunk[lastCorrectType] = 1.0

    if foundPredCnt > 0:
        precision = 1.0 * correctChunkCnt / foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 1.0 * correctChunkCnt / foundCorrectCnt
    else:
        recall = 0

    if (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall


def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
    """
	Compute the accuracy based on the whole predictions of
	given sentence, including slot and intent.
	"""
    total_count, correct_count = 0.0, 0.0
    for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot,
                                                  pred_intent, real_intent):

        if p_slot == r_slot and set(p_intent) == set(r_intent):
            correct_count += 1.0
        total_count += 1.0

    return 1.0 * correct_count / total_count
