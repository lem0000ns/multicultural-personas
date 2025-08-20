import json
from utils.configs import configs

hm_resource = {"afar": 0, "balochi": 0, "faroese": 0, "fijian": 0, "hiligaynon": 0, "kirundi": 0, "papiamento": 0, "pashto": 0, "samoan": 0, "tongan": 0, "tswana": 0, "wolof": 0, "english": 1, "chinese": 1, "spanish": 1, "german": 1, "arabic": 1, "hebrew": 1, "hindi": 1, "hungarian": 1, "japanese": 1, "korean": 1, "russian": 1}

def analyze(config_name, type):
    with open(f"personaData/{type}/{config_name}-results.json", "r") as f:
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

        if entry["vanilla_issue"] and entry["persona_issue"]:
            continue
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
    
    # Calculate metrics
    hm_vanilla_accuracy = hm_vanilla_correct / hm_total_ground_truth if hm_total_ground_truth > 0 else 0
    hm_persona_accuracy = hm_persona_correct / hm_total_ground_truth if hm_total_ground_truth > 0 else 0
    hm_vanilla_avg_confidence = hm_vanilla_confidence / hm_total_ground_truth if hm_total_ground_truth > 0 else 0
    hm_persona_avg_confidence = hm_persona_confidence / hm_total_ground_truth if hm_total_ground_truth > 0 else 0
    
    lr_vanilla_accuracy = lr_vanilla_correct / lr_total_ground_truth if lr_total_ground_truth > 0 else 0
    lr_persona_accuracy = lr_persona_correct / lr_total_ground_truth if lr_total_ground_truth > 0 else 0
    lr_vanilla_avg_confidence = lr_vanilla_confidence / lr_total_ground_truth if lr_total_ground_truth > 0 else 0
    lr_persona_avg_confidence = lr_persona_confidence / lr_total_ground_truth if lr_total_ground_truth > 0 else 0
    
    # Eye test metrics
    hm_vanilla_eye_accuracy = hm_vanilla_eye_test / hm_total_eye_test if hm_total_eye_test > 0 else 0
    hm_persona_eye_accuracy = hm_persona_eye_test / hm_total_eye_test if hm_total_eye_test > 0 else 0
    hm_vanilla_eye_avg_confidence = hm_vanilla_eye_test_confidence / hm_total_vanilla_eye_test if hm_total_vanilla_eye_test > 0 else 0
    hm_persona_eye_avg_confidence = hm_persona_eye_test_confidence / hm_total_persona_eye_test if hm_total_persona_eye_test > 0 else 0
    
    lr_vanilla_eye_accuracy = lr_vanilla_eye_test / lr_total_eye_test if lr_total_eye_test > 0 else 0
    lr_persona_eye_accuracy = lr_persona_eye_test / lr_total_eye_test if lr_total_eye_test > 0 else 0
    lr_vanilla_eye_avg_confidence = lr_vanilla_eye_test_confidence / lr_total_vanilla_eye_test if lr_total_vanilla_eye_test > 0 else 0
    lr_persona_eye_avg_confidence = lr_persona_eye_test_confidence / lr_total_persona_eye_test if lr_total_persona_eye_test > 0 else 0
    
    # Overall metrics
    total_ground_truth = hm_total_ground_truth + lr_total_ground_truth
    total_eye_test = hm_total_eye_test + lr_total_eye_test
    
    vanilla_correct = hm_vanilla_correct + lr_vanilla_correct
    persona_correct = hm_persona_correct + lr_persona_correct
    vanilla_confidence = hm_vanilla_confidence + lr_vanilla_confidence
    persona_confidence = hm_persona_confidence + lr_persona_confidence
    
    vanilla_eye_test = hm_vanilla_eye_test + lr_vanilla_eye_test
    persona_eye_test = hm_persona_eye_test + lr_persona_eye_test
    vanilla_eye_test_confidence = hm_vanilla_eye_test_confidence + lr_vanilla_eye_test_confidence
    persona_eye_test_confidence = hm_persona_eye_test_confidence + lr_persona_eye_test_confidence
    total_vanilla_eye_test = hm_total_vanilla_eye_test + lr_total_vanilla_eye_test
    total_persona_eye_test = hm_total_persona_eye_test + lr_total_persona_eye_test
    
    overall_vanilla_accuracy = vanilla_correct / total_ground_truth if total_ground_truth > 0 else 0
    overall_persona_accuracy = persona_correct / total_ground_truth if total_ground_truth > 0 else 0
    overall_vanilla_avg_confidence = vanilla_confidence / total_ground_truth if total_ground_truth > 0 else 0
    overall_persona_avg_confidence = persona_confidence / total_ground_truth if total_ground_truth > 0 else 0
    
    overall_vanilla_eye_accuracy = vanilla_eye_test / total_eye_test if total_eye_test > 0 else 0
    overall_persona_eye_accuracy = persona_eye_test / total_eye_test if total_eye_test > 0 else 0
    overall_vanilla_eye_avg_confidence = vanilla_eye_test_confidence / total_vanilla_eye_test if total_vanilla_eye_test > 0 else 0
    overall_persona_eye_avg_confidence = persona_eye_test_confidence / total_persona_eye_test if total_persona_eye_test > 0 else 0
    
    # Create results dictionary
    results = {
        "config_name": config_name,
        "type": type,
        "high_medium_resource": {
            "ground_truth": {
                "total": hm_total_ground_truth,
                "vanilla_correct": hm_vanilla_correct,
                "persona_correct": hm_persona_correct,
                "vanilla_accuracy": hm_vanilla_accuracy,
                "persona_accuracy": hm_persona_accuracy,
                "vanilla_avg_confidence": hm_vanilla_avg_confidence,
                "persona_avg_confidence": hm_persona_avg_confidence
            },
            "eye_test": {
                "total": hm_total_eye_test,
                "vanilla_correct": hm_vanilla_eye_test,
                "persona_correct": hm_persona_eye_test,
                "vanilla_accuracy": hm_vanilla_eye_accuracy,
                "persona_accuracy": hm_persona_eye_accuracy,
                "vanilla_avg_confidence": hm_vanilla_eye_avg_confidence,
                "persona_avg_confidence": hm_persona_eye_avg_confidence
            }
        },
        "low_resource": {
            "ground_truth": {
                "total": lr_total_ground_truth,
                "vanilla_correct": lr_vanilla_correct,
                "persona_correct": lr_persona_correct,
                "vanilla_accuracy": lr_vanilla_accuracy,
                "persona_accuracy": lr_persona_accuracy,
                "vanilla_avg_confidence": lr_vanilla_avg_confidence,
                "persona_avg_confidence": lr_persona_avg_confidence
            },
            "eye_test": {
                "total": lr_total_eye_test,
                "vanilla_correct": lr_vanilla_eye_test,
                "persona_correct": lr_persona_eye_test,
                "vanilla_accuracy": lr_vanilla_eye_accuracy,
                "persona_accuracy": lr_persona_eye_accuracy,
                "vanilla_avg_confidence": lr_vanilla_eye_avg_confidence,
                "persona_avg_confidence": lr_persona_eye_avg_confidence
            }
        },
        "overall": {
            "ground_truth": {
                "total": total_ground_truth,
                "vanilla_correct": vanilla_correct,
                "persona_correct": persona_correct,
                "vanilla_accuracy": overall_vanilla_accuracy,
                "persona_accuracy": overall_persona_accuracy,
                "vanilla_avg_confidence": overall_vanilla_avg_confidence,
                "persona_avg_confidence": overall_persona_avg_confidence
            },
            "eye_test": {
                "total": total_eye_test,
                "vanilla_correct": vanilla_eye_test,
                "persona_correct": persona_eye_test,
                "vanilla_accuracy": overall_vanilla_eye_accuracy,
                "persona_accuracy": overall_persona_eye_accuracy,
                "vanilla_avg_confidence": overall_vanilla_eye_avg_confidence,
                "persona_avg_confidence": overall_persona_eye_avg_confidence
            }
        }
    }
    
    # Save results to JSON file
    output_file = f"personaData/{type}/{config_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
            

if __name__ == "__main__":
    try:
        for config_name in configs:
            print(f"\n{'='*60}")
            print(f"ANALYZING CONFIG: {config_name}")
            print(f"{'='*60}")
            
            # Analyze agnostic type
            print(f"\nAnalyzing agnostic type...")
            analyze(config_name, "agnostic")
            
            # Analyze specific type
            print(f"\nAnalyzing specific type...")
            analyze(config_name, "specific")
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")