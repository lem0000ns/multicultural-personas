import json

def normalize_answer(answer):
    """Normalize answer to boolean for comparison."""
    if isinstance(answer, str):
        return answer.lower() == "true"
    return bool(answer)

def verify_country(country: str, iteration: int):
    with open("./results/ling/i5/persona_Hard.jsonl", "r") as f:
        data = [json.loads(line) for line in f if line.strip() and "Accuracy" not in line]
    correct = 0
    total = 0
    for i in range(0, len(data), 4):
        if data[i]["country"] != country or data[i]["iteration"] != iteration:
            continue
        is_correct = True
        for j in range(i, i+4):
            correct_answer = normalize_answer(data[j]["correct_answer"])
            persona_answer = normalize_answer(data[j]["persona_answer"])
            if correct_answer != persona_answer:
                is_correct = False
                break
        if is_correct:
            correct += 1
        total += 1
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {correct/total}")

if __name__ == "__main__":
    verify_country("Czech Republic", 1)