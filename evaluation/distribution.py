import json
import argparse
from tools.utils import country_to_continent, modes_list, modes_list_p1, modes_list_p2

def process_data(type, difficulty, iter, mode):
    try:
        file_name = f"../results/{mode[-2:]}/{mode[:-3]}/i{iter}/{type}_{difficulty}.jsonl" if type != "vanilla" else f"../results/vanilla/vanilla_{difficulty}.jsonl"
        with open(file_name, "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines[:-1]]
            accuracy = (float(lines[-1].split(" ")[-1].strip()) * 100)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None
    
    stats = {}
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
    return stats, accuracy

def country_distribution(mode):
    for iter in range(1, 2):
        try:
            # Process persona-prompted culturalbench-easy
            persona_easy_stats, persona_easy_accuracy = process_data("persona", "Easy", iter, mode)

            # Process vanilla culturalbench-easy
            vanilla_easy_stats, vanilla_easy_accuracy = process_data("vanilla", "Easy", iter, "vanilla")
            
            # Process persona-prompted culturalbench-hard
            persona_hard_stats, persona_hard_accuracy = process_data("persona", "Hard", iter, mode)
            
            # Process vanilla culturalbench-hard
            vanilla_hard_stats, vanilla_hard_accuracy = process_data("vanilla", "Hard", iter, "vanilla")
            
            with open(f"../results/{mode[-2:]}/{mode[:-3]}/i{iter}/country_distribution.json", "w") as f:
                json.dump({
                    "vanilla": {
                        "easy": {"Accuracy": vanilla_easy_accuracy, "Regions": vanilla_easy_stats or {}}, 
                        "hard": {"Accuracy": vanilla_hard_accuracy, "Regions": vanilla_hard_stats or {}}
                    }, 
                    "persona": {
                        "easy": {"Accuracy": persona_easy_accuracy, "Regions": persona_easy_stats or {}}, 
                        "hard": {"Accuracy": persona_hard_accuracy, "Regions": persona_hard_stats or {}}
                    },
                    "net_persona": {
                        "easy": {
                            "Net Accuracy": persona_easy_accuracy - vanilla_easy_accuracy,
                            "Regions": {
                                region: persona_easy_stats[region] - vanilla_easy_stats[region]
                                for region in (persona_easy_stats or {}).keys()
                            }
                        } if persona_easy_stats and vanilla_easy_stats else {},
                        "hard": {
                            "Net Accuracy": persona_hard_accuracy - vanilla_hard_accuracy,
                            "Regions": {
                                region: persona_hard_stats[region] - vanilla_hard_stats[region]
                                for region in (persona_hard_stats or {}).keys()
                            }
                        } if persona_hard_stats and vanilla_hard_stats else {}
                    }
                }, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Error processing {mode}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=False, choices=["p1", "p2", "both"])
    parser.add_argument("--mode", type=str, nargs="+", required=False, choices=modes_list + ["all"], default=["all"])
    args = parser.parse_args()

    try:
        # if user specified prompt, run all modes associated with that prompt
        if args.prompt == "p1" or args.prompt == "both":
            for mode in modes_list_p1:
                country_distribution(mode=mode)
        if args.prompt == "p2" or args.prompt == "both":
            for mode in modes_list_p2:
                country_distribution(mode=mode)

        # run remaining modes
        for mode in args.mode if args.mode[0] != "all" else modes_list:
            if (args.prompt == "p1" or args.prompt == "both") and mode[-2:] == "p1":
                continue
            if (args.prompt == "p2" or args.prompt == "both") and mode[-2:] == "p2":
                continue
            country_distribution(mode=mode)
    except Exception as e:
        print(e)
        exit(1)