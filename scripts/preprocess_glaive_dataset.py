import re
import json


with open("./datasets/glaive/glaive-function-calling-v2.json", "r") as f:
    data = json.load(f)


def process_sample(sample):
    system_text = sample.get("system", "")
    chat_text = sample.get("chat", "")

    # Extract tools
    tools_match = re.findall(
        r'"name": "(.*?)",.*?"description": "(.*?)"', system_text, re.S
    )
    if not tools_match:
        return None  # Skip samples without tools
    tools = [{"name": name, "description": desc} for name, desc in tools_match]

    # Extract function calls and corresponding instructions
    function_calls = re.findall(r'<functioncall> {"name": "(.*?)"', chat_text)
    if not function_calls:
        return None  # Skip samples without function calls

    # Extract ASSISTANT responses and filter for valid instructions
    assistant_responses = re.findall(
        r"ASSISTANT: (.*?)<\|endoftext\|>", chat_text, re.S
    )
    instructions = []
    for call, response in zip(function_calls, assistant_responses):
        # Find the user prompt just before this ASSISTANT response
        user_prompt_match = re.search(
            rf'USER: (.*?)\n.*?<functioncall> {{.*?"{call}".*?}}', chat_text, re.S
        )
        if user_prompt_match:
            instructions.append(
                {
                    "instruction": user_prompt_match.group(1).strip(),
                    "tools": tools,
                    "used_tools": [call],  # Single tool for this instruction
                }
            )

    return instructions


from tqdm import tqdm

results = []
for sample in tqdm(data, desc="Processing GLAIVE samples"):
    processed = process_sample(sample)
    if processed:
        results.extend(processed)

with open("./datasets/glaive/output.json", "w") as f:
    json.dump(results, f, indent=2)
