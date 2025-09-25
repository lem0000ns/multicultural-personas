instruction_1 = "Have expertise directly RELEVANT to the question content."
instruction_2 = "Provide cultural or linguistic context that enhances understanding."
instruction_3 = "Guide the model to analyze the question carefully and choose the most accurate option."
instruction_4 = "Contain only the persona description — no extra explanations, formatting, or translations."
instruction_5 = "Be direct, concise, and focused on reasoning for multiple-choice or true-false questions."
eng_second_person = "Always start with 'You are'... followed by the persona description."
ling_second_person = "Always start with second-person pronoun followed by the persona description."

def build_prompt(instructions: list[str], second_person_note: str) -> str:
    return """You are an expert in crafting {language} personas that will be used as a system prompt to a language model in answering a multiple-choice or true/false question accurately. You will be given the question and the relevant country that the question content is related to. The language of the persona description MUST BE in {language}.

    Your task is to generate a persona that MUST:""" + "\n".join([f"{i + 1}. {instruction}" for i, instruction in enumerate(instructions)]) + """
    

    # IMPORTANT \n#
    """ + second_person_note

system_prompts = {
    "eng": 
        build_prompt(instructions=[instruction_1, instruction_2, instruction_3, instruction_4, instruction_5], second_person_note=eng_second_person),
    
    "eng_no_expertise": 
        build_prompt(instructions=[instruction_2, instruction_3, instruction_4, instruction_5], second_person_note=eng_second_person),
    
    "eng_no_cultural": 
        build_prompt(instructions=[instruction_1, instruction_3, instruction_4, instruction_5], second_person_note=eng_second_person),
    
    "eng_no_reasoning":
        build_prompt(instructions=[instruction_1, instruction_2, instruction_4, instruction_5], second_person_note=eng_second_person),

    "eng_no_direct":
        build_prompt(instructions=[instruction_1, instruction_2, instruction_3, instruction_4], second_person_note=eng_second_person),
    
    "ling":
        build_prompt(instructions=[instruction_1, instruction_2, instruction_3, instruction_4, instruction_5], second_person_note=ling_second_person),
    
    "ling_no_expertise":
        build_prompt(instructions=[instruction_2, instruction_3, instruction_4, instruction_5], second_person_note=ling_second_person),
    
    "ling_no_cultural":
        build_prompt(instructions=[instruction_1, instruction_3, instruction_4, instruction_5], second_person_note=ling_second_person),
    
    "ling_no_reasoning":
        build_prompt(instructions=[instruction_1, instruction_2, instruction_4, instruction_5], second_person_note=ling_second_person),
    
    "ling_no_direct":
        build_prompt(instructions=[instruction_1, instruction_2, instruction_3, instruction_4], second_person_note=ling_second_person),

}