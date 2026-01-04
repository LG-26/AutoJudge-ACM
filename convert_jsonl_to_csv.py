import json
import csv

input_file = "data/problems_data.jsonl"   
output_file = "data/problems.csv"

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

fieldnames = data[0].keys()

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print("Conversion completed: problems.csv created")