import json
import sys
from utils import country_to_continent

def process_data(type, difficulty, mode):
    try:
        with open(f"{mode}/{type}_{difficulty}.jsonl", "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines[:-1]]
            accuracy = lines[-1]
    except Exception as e:
        print(f"Error processing {mode}/{type}_{difficulty}.jsonl: {e}")
        return None
    
    stats = {}
    stats["Accuracy"] = accuracy
    correct_counts = {}
    for item in data:
        country = item["country"]
        continent = country_to_continent[country]
        if continent not in correct_counts:
            correct_counts[continent] = {"total": 1, "correct": 1 if str(item[type + "_answer"]).upper() == str(item["correct_answer"]).upper() else 0}
        else:
            correct_counts[continent]["total"] += 1
            correct_counts[continent]["correct"] += 1 if str(item[type + "_answer"]).upper() == str(item["correct_answer"]).upper() else 0
    for continent in correct_counts:
        stats[continent] = correct_counts[continent]["correct"] / correct_counts[continent]["total"]
    return stats

def country_distribution(mode):
    # Process persona-prompted culturalbench-easy
    persona_easy_stats = process_data("persona", "Easy", mode)

    # Process vanilla culturalbench-easy
    vanilla_easy_stats = process_data("vanilla", "Easy", "vanilla")
    
    # Process persona-prompted culturalbench-hard
    persona_hard_stats = process_data("persona", "Hard", mode)
    
    # Process vanilla culturalbench-hard
    vanilla_hard_stats = process_data("vanilla", "Hard", "vanilla")
    
    with open(f"{mode}/country_distribution.json", "w") as f:
        json.dump({
            "vanilla": {
                "easy": vanilla_easy_stats or {}, 
                "hard": vanilla_hard_stats or {}
            }, 
            "persona": {
                "easy": persona_easy_stats or {}, 
                "hard": persona_hard_stats or {}
            },
            "net_persona": {
                "easy": {
                    region: persona_easy_stats[region] - vanilla_easy_stats[region]
                    for region in (persona_easy_stats or {}).keys() if region != "Accuracy"
                } if persona_easy_stats and vanilla_easy_stats else {},
                "hard": {
                    region: persona_hard_stats[region] - vanilla_hard_stats[region]
                    for region in (persona_hard_stats or {}).keys() if region != "Accuracy"
                } if persona_hard_stats and vanilla_hard_stats else {}
            }
        }, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eng"
    if mode not in ["detailed_eng", "eng", "ling", "brief_eng", "detailed_ling", "brief_ling"]:
        print("Invalid mode. Valid modes are: " + ", ".join(["detailed_eng", "eng", "ling", "brief_eng", "detailed_ling", "brief_ling"]))
        exit(1)
    country_distribution(mode)