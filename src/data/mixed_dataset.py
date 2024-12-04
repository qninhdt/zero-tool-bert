import os
import json
import torch
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MixedDataset(Dataset):
    def __init__(self, bert_model, stage, anno_file, tool_capacity, seed):
        self.stage = stage
        self.seed = seed
        self.tool_capacity = tool_capacity
        self.tools, self.samples = self.load_data(anno_file)
        self.tool_ids = list(self.tools.keys())
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    def load_data(self, anno_file):
        with open(anno_file, "r") as f:
            data = json.load(f)
        tools = data["tools"]
        samples = data["samples"]

        tools = {tool["id"]: tool for tool in tools}

        return tools, samples

    def encode_text(self, text, padding=True):
        if padding:
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
            )
        else:
            inputs = self.tokenizer(
                text,
                max_length=128,
                truncation=True,
            )
        ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        return ids, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inst = sample["instruction"]
        inst_ids, inst_mask = self.encode_text(inst)

        if self.stage == "train":
            tool_id = random.choice(sample["tools"])
            tool_desc = self.tools[tool_id]["description"]
            tool_desc_ids, tool_desc_mask = self.encode_text(tool_desc)

            return {
                "inst_ids": inst_ids,
                "inst_mask": inst_mask,
                "tool_ids": tool_desc_ids,
                "tool_mask": tool_desc_mask,
            }
        else:
            # for testing, we sample a random set of tools + the correct tool, size = tool_capacity
            # wrong tools are sampled randomly from self.tools
            correct_tools = sample["tools"]

            random.seed(self.seed + idx)
            wrong_tools = random.sample(
                [tool for tool in self.tool_ids if tool not in correct_tools],
                self.tool_capacity - len(correct_tools),
            )

            correct_tool_mask = torch.tensor(
                [1] * len(correct_tools)
                + [0] * (self.tool_capacity - len(correct_tools)),
                dtype=torch.bool,
            )

            tools = correct_tools + wrong_tools
            tool_ids, tool_mask = self.encode_text(
                [self.tools[tool_id]["description"] for tool_id in tools]
            )

            return {
                "inst_ids": inst_ids,
                "inst_mask": inst_mask,
                "tool_ids": tool_ids,
                "tool_mask": tool_mask,
                "correct_tool_mask": correct_tool_mask,
            }
