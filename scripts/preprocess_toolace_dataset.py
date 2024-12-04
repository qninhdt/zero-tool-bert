import json
import re

with open("./datasets/toolace/data.json") as f:
    data = json.load(f)


def process_sample(sample):
    system_message = sample["system"]
    functions_match = re.findall(
        r'\{"name": "(.*?)", "description": "(.*?)"', system_message
    )
    functions = [
        {"name": name.replace(" ", "_").replace("/", "_"), "description": desc}
        for name, desc in functions_match
    ]

    # Process the conversation to extract instructions and function calls
    conversations = sample["conversations"]
    results = []

    for i in range(len(conversations)):
        entry = conversations[i]

        # Check if the entry is from the user and the next entry is an assistant function call
        if (
            entry["from"] == "user"
            and i + 1 < len(conversations)
            and conversations[i + 1]["from"] == "assistant"
        ):
            function_call = conversations[i + 1]["value"]
            if re.match(r"\[.*\]", function_call):  # Ensure it contains a function call
                # Extract full tool names (including spaces) from the function call
                used_tools = [
                    tool.replace(" ", "_").replace("/", "_")
                    for tool in re.findall(r"\[([^\(]+)\(", function_call)
                ]

                if len(used_tools) == 0:
                    continue

                results.append(
                    {
                        "instruction": entry["value"],
                        "tools": functions,
                        "used_tools": used_tools,
                    }
                )

    return results


from tqdm import tqdm

results = []
for sample in tqdm(data, desc="Processing ToolACE samples"):
    processed = process_sample(sample)
    if processed:
        results.extend(processed)

with open("./datasets/toolace/output.json", "w") as f:
    json.dump(results, f, indent=2)
