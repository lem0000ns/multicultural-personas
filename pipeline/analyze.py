import json

hm_resource = {"afar": 0, "balochi": 0, "faroese": 0, "fijian": 0, "hiligaynon": 0, "kirundi": 0, "papiamento": 0, "pashto": 0, "samoan": 0, "tongan": 0, "tswana": 0, "wolof": 0, "english": 1, "chinese": 1, "spanish": 1, "german": 1, "arabic": 1, "hebrew": 1, "hindi": 1, "hungarian": 1, "japanese": 1, "korean": 1, "russian": 1}

def analyze(type):
    with open(f"personaData/{type}-j.json", "r") as f:
        data = json.load(f)
    
    # Initialize counters for high/medium resource languages
    hm_vanilla_correct = hm_persona_correct = hm_total_ground_truth = hm_vanilla_eye_test = hm_persona_eye_test = hm_total_eye_test = 0
    hm_vanilla_confidence = hm_persona_confidence = hm_vanilla_eye_test_confidence = hm_persona_eye_test_confidence = 0
    hm_total_vanilla_eye_test = hm_total_persona_eye_test = 0
    
    # Initialize counters for low resource languages
    lr_vanilla_correct = lr_persona_correct = lr_total_ground_truth = lr_vanilla_eye_test = lr_persona_eye_test = lr_total_eye_test = 0
    lr_vanilla_confidence = lr_persona_confidence = lr_vanilla_eye_test_confidence = lr_persona_eye_test_confidence = 0
    lr_total_vanilla_eye_test = lr_total_persona_eye_test = 0
    
    for entry in data:
        # Determine if this is a high/medium resource language
        language = entry.get("language", "").lower()
        is_hm_resource = hm_resource.get(language, 0) == 1
        
        if entry["ground_truth"] != "None":
            if is_hm_resource:
                hm_total_ground_truth += 1
                if entry["vanilla_comparison"]["correct"] == "yes":
                    hm_vanilla_correct += 1
                if entry["persona_comparison"]["correct"] == "yes":
                    hm_persona_correct += 1
                hm_vanilla_confidence += int(entry["vanilla_comparison"]["confidence"])
                hm_persona_confidence += int(entry["persona_comparison"]["confidence"])
            else:
                lr_total_ground_truth += 1
                if entry["vanilla_comparison"]["correct"] == "yes":
                    lr_vanilla_correct += 1
                if entry["persona_comparison"]["correct"] == "yes":
                    lr_persona_correct += 1
                lr_vanilla_confidence += int(entry["vanilla_comparison"]["confidence"])
                lr_persona_confidence += int(entry["persona_comparison"]["confidence"])
        else:
            if is_hm_resource:
                hm_total_eye_test += 1
                if entry["comparison"]["verdict"] == "Answer 1":
                    hm_vanilla_eye_test += 1
                    hm_vanilla_eye_test_confidence += int(entry["comparison"]["confidence"])
                    hm_total_vanilla_eye_test += 1
                if entry["comparison"]["verdict"] == "Answer 2":
                    hm_persona_eye_test += 1
                    hm_persona_eye_test_confidence += int(entry["comparison"]["confidence"])
                    hm_total_persona_eye_test += 1
            else:
                lr_total_eye_test += 1
                if entry["comparison"]["verdict"] == "Answer 1":
                    lr_vanilla_eye_test += 1
                    lr_vanilla_eye_test_confidence += int(entry["comparison"]["confidence"])
                    lr_total_vanilla_eye_test += 1
                if entry["comparison"]["verdict"] == "Answer 2":
                    lr_persona_eye_test += 1
                    lr_persona_eye_test_confidence += int(entry["comparison"]["confidence"])
                    lr_total_persona_eye_test += 1
    
    print("-" * 50)
    print(f"COMPARISON RESULTS FOR {type.upper()}")
    print("-" * 50)
    
    # High/Medium Resource Language Results
    print("HIGH/MEDIUM RESOURCE LANGUAGES:")
    print(f"Total ground truth: {hm_total_ground_truth}")
    print(f"Total eye test: {hm_total_eye_test}\n")
    
    if hm_total_ground_truth > 0:
        print(f"Vanilla correct rate: {hm_vanilla_correct / hm_total_ground_truth:.2f}")
        print(f"Average confidence: {hm_vanilla_confidence / hm_total_ground_truth:.2f}")
        print(f"Persona correct rate: {hm_persona_correct / hm_total_ground_truth:.2f}")
        print(f"Average confidence: {hm_persona_confidence / hm_total_ground_truth:.2f}\n")

    if hm_total_eye_test > 0:
        print(f"Vanilla eye test correct rate: {hm_vanilla_eye_test / hm_total_eye_test:.2f}")
        if hm_total_vanilla_eye_test > 0:
            print(f"Average confidence: {hm_vanilla_eye_test_confidence / hm_total_vanilla_eye_test:.2f}")
        print(f"Persona eye test correct rate: {hm_persona_eye_test / hm_total_eye_test:.2f}")
        if hm_total_persona_eye_test > 0:
            print(f"Average confidence: {hm_persona_eye_test_confidence / hm_total_persona_eye_test:.2f}")
    
    print("\n" + "-" * 50)
    
    # Low Resource Language Results
    print("LOW RESOURCE LANGUAGES:")
    print(f"Total ground truth: {lr_total_ground_truth}")
    print(f"Total eye test: {lr_total_eye_test}\n")
    
    if lr_total_ground_truth > 0:
        print(f"Vanilla correct rate: {lr_vanilla_correct / lr_total_ground_truth:.2f}")
        print(f"Average confidence: {lr_vanilla_confidence / lr_total_ground_truth:.2f}")
        print(f"Persona correct rate: {lr_persona_correct / lr_total_ground_truth:.2f}")
        print(f"Average confidence: {lr_persona_confidence / lr_total_ground_truth:.2f}\n")

    if lr_total_eye_test > 0:
        print(f"Vanilla eye test correct rate: {lr_vanilla_eye_test / lr_total_eye_test:.2f}")
        if lr_total_vanilla_eye_test > 0:
            print(f"Average confidence: {lr_vanilla_eye_test_confidence / lr_total_vanilla_eye_test:.2f}")
        print(f"Persona eye test correct rate: {lr_persona_eye_test / lr_total_eye_test:.2f}")
        if lr_total_persona_eye_test > 0:
            print(f"Average confidence: {lr_persona_eye_test_confidence / lr_total_persona_eye_test:.2f}")
    
    print("\n" + "-" * 50)
    
    # Overall Results
    total_ground_truth = hm_total_ground_truth + lr_total_ground_truth
    total_eye_test = hm_total_eye_test + lr_total_eye_test
    
    print("OVERALL RESULTS:")
    print(f"Total ground truth: {total_ground_truth}")
    print(f"Total eye test: {total_eye_test}\n")
    
    if total_ground_truth > 0:
        vanilla_correct = hm_vanilla_correct + lr_vanilla_correct
        persona_correct = hm_persona_correct + lr_persona_correct
        vanilla_confidence = hm_vanilla_confidence + lr_vanilla_confidence
        persona_confidence = hm_persona_confidence + lr_persona_confidence
        
        print(f"Vanilla correct rate: {vanilla_correct / total_ground_truth:.2f}")
        print(f"Average confidence: {vanilla_confidence / total_ground_truth:.2f}")
        print(f"Persona correct rate: {persona_correct / total_ground_truth:.2f}")
        print(f"Average confidence: {persona_confidence / total_ground_truth:.2f}\n")

    if total_eye_test > 0:
        vanilla_eye_test = hm_vanilla_eye_test + lr_vanilla_eye_test
        persona_eye_test = hm_persona_eye_test + lr_persona_eye_test
        vanilla_eye_test_confidence = hm_vanilla_eye_test_confidence + lr_vanilla_eye_test_confidence
        persona_eye_test_confidence = hm_persona_eye_test_confidence + lr_persona_eye_test_confidence
        total_vanilla_eye_test = hm_total_vanilla_eye_test + lr_total_vanilla_eye_test
        total_persona_eye_test = hm_total_persona_eye_test + lr_total_persona_eye_test
        
        print(f"Vanilla eye test correct rate: {vanilla_eye_test / total_eye_test:.2f}")
        if total_vanilla_eye_test > 0:
            print(f"Average confidence: {vanilla_eye_test_confidence / total_vanilla_eye_test:.2f}")
        print(f"Persona eye test correct rate: {persona_eye_test / total_eye_test:.2f}")
        if total_persona_eye_test > 0:
            print(f"Average confidence: {persona_eye_test_confidence / total_persona_eye_test:.2f}")

if __name__ == "__main__":
    try:
        analyze("ag")
        analyze("sp")
    except Exception as e:
        print(f"Error analyzing {type}: {e}")