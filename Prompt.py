
data_template = """
[Instruction]
you are an expert of spoken language understanding, I need you to perform intent detection and slot filling for given utterance. \n

[Input]
utterance: {utterance} 

[Response]
intent: {intent}
entity_slot: {entity_slots}
"""

data_template_sub =  """
[Instruction]
you are an expert of spoken language understanding, I need you to perform intent detection and slot filling for given utterance. \n

[Input]
utterance: {utterance} 
sub_utterance: {sub_utterance} 

[Response]
intent: {intent}
entity_slot: {entity_slots}
"""

test_template =  """
[Instruction]
you are an expert of spoken language understanding, I need you to perform intent detection and slot filling for given utterance. \n

[Input]
utterance: {utterance} 

[Response]

"""