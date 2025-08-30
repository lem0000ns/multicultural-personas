import argparse
import json
import matplotlib.pyplot as plt
from tools.utils import modes_list

def display_net_accuracy(metric, order_by):
    x_data = modes_list
    y_data = []
    for mode in modes_list:
        with open(f"../results/{mode}/country_distribution.json", "r") as f:
            data = json.load(f)
            if order_by == "both":
                average = (data["net_persona"]["easy"]["Net Accuracy"] + data["net_persona"]["hard"]["Net Accuracy"]) / 2
                y_data.append(average)
            else:
                y_data.append(data["net_persona"][order_by]["Net Accuracy"])

    # Sort y_data and x_data in increasing order
    sorted_pairs = sorted(zip(y_data, x_data))
    y_data, x_data = zip(*sorted_pairs)
    
    plt.figure(figsize=(12, 6))
    plt.bar(x_data, y_data)
    plt.xlabel("Mode")
    plt.ylabel(f"{'Average ' if order_by == 'both' else ''}Net Accuracy")
    plt.title("Net Accuracy by Mode for " + order_by.capitalize() + (" difficulty" if order_by != "both" else ""))
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    
    plt.savefig(f"../visualizations/{metric}/{order_by}.png", dpi=300, bbox_inches='tight')

def display_region_accuracy(order_by):
    print("Not implemented")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True, choices=["net_accuracy", "region_accuracy"])
    parser.add_argument("--order_by", type=str, required=True, choices=["hard", "easy", "both", "all"])
    args = parser.parse_args()
    
    if args.metric == "net_accuracy":
        if args.order_by == "all":
            display_net_accuracy(args.metric, "hard")
            display_net_accuracy(args.metric, "easy")
            display_net_accuracy(args.metric, "both")
        else:
            display_net_accuracy(args.metric, args.order_by)
    elif args.metric == "region_accuracy":
        display_region_accuracy(args.order_by)