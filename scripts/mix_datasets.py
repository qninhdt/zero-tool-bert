import json
from tqdm import tqdm

DATASETS = ["apigen", "glaive", "toolace"]

data = []

for dataset_name in DATASETS:
    with open(f"./datasets/{dataset_name}/output.json") as f:
        subdata = json.load(f)

    subdata = [{**item, "source": dataset_name} for item in subdata]

    data.extend(subdata)

# filter all samples with used functions not in tool list
new_data = []

for sample in data:
    tools = [tool["name"] for tool in sample["tools"]]

    if any([used_tool not in tools for used_tool in sample["used_tools"]]):
        continue

    new_data.append(sample)

data = new_data

print("Number of samples:", len(data))


def generate_dataset(data):
    tools = {}
    dataset = []

    for sample in tqdm(data, desc="Processing samples"):

        for tool in sample["tools"]:
            if tool["name"] in tools:
                continue

            tools[tool["name"]] = {
                "name": tool["name"],
                "description": tool["description"],
                "id": len(tools),
                "source": sample["source"],
            }

        used_tools = []

        for tool_name in sample["used_tools"]:
            used_tools.append(tools[tool_name]["id"])

        new_sample = {
            "instruction": sample["instruction"],
            "tools": used_tools,
            "source": sample["source"],
        }

        dataset.append(new_sample)

    return {"tools": list(tools.values()), "samples": dataset}


from collections import defaultdict


def count(data):
    used_tools_count = defaultdict(int)

    for item in data:
        used_tools_count[len(item["used_tools"])] += 1

    return used_tools_count


print("Used tools count in dataset:")
for k, v in count(data).items():
    print(f"{k} used tools: {v} samples")

from random import shuffle, seed

# split dataset based on used tools count
# for one and two used tools we will use 80-20 split
# for three used tools we will all samples in test set
one_two = [item for item in data if len(item["used_tools"]) in [1, 2]]
other = [item for item in data if len(item["used_tools"]) > 2]

seed(42)
shuffle(one_two)

seed(42)
shuffle(other)

train_samples = one_two[: int(len(one_two) * 0.8)]
test_samples = one_two[int(len(one_two) * 0.8) :] + other

print("Train samples count:", len(train_samples))
print("Test samples count:", len(test_samples))

# count(train_samples), count(test_samples)
print("Train samples count based on used tools count:")
for k, v in count(train_samples).items():
    print(f"{k} used tools: {v} samples")

print("Test samples count based on used tools count:")
for k, v in count(test_samples).items():
    print(f"{k} used tools: {v} samples")

train_dataset = generate_dataset(train_samples)
test_dataset = generate_dataset(test_samples)

seed(42)
shuffle(train_dataset["samples"])

seed(42)
shuffle(test_dataset["samples"])

print("Number of tools in train dataset:", len(train_dataset["tools"]))
print("Number of samples in train dataset:", len(train_dataset["samples"]))

print("Number of tools in test dataset:", len(test_dataset["tools"]))
print("Number of samples in test dataset:", len(test_dataset["samples"]))

import os

os.makedirs("./datasets/mixed", exist_ok=True)

with open("./datasets/mixed/train.json", "w") as f:
    json.dump(train_dataset, f, indent=2)

with open("./datasets/mixed/test.json", "w") as f:
    json.dump(test_dataset, f, indent=2)
