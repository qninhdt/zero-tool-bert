import json

with open("./datasets/apigen/xlam_function_calling_60k.json", "r") as f:
    data = json.load(f)

results = []
max_tool_usage = 0

from tqdm import tqdm

for sample in tqdm(data, desc="Processing APIGen samples"):
    used_tools = list(set([x["name"] for x in json.loads(sample["answers"])]))

    if len(used_tools) == 0:
        continue

    tools = [
        {"name": x["name"].replace(".", "_"), "description": x["description"]}
        for x in json.loads(sample["tools"])
    ]

    result = {"instruction": sample["query"], "tools": tools, "used_tools": used_tools}

    results.append(result)

with open("./datasets/apigen/output.json", "w") as f:
    json.dump(results, f, indent=4)
