
import os
from typing import Dict, List
from torch.utils.data import Dataset

class ProtocolJsonDataset(Dataset):
    """Paired dataset of protocol text (.txt) and extracted actions (.json)."""
    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 max_len: int = 1024,
                 automatable_actions: List[str] = None):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.automatable_actions = automatable_actions or [
            "pipette","dispense","aspirate","mix","shake",
            "incubate","heat","cool","centrifuge","wait"
        ]

        for fname in os.listdir(data_dir):
            if not fname.endswith(".txt"):
                continue
            base = fname[:-4]
            txt_path = os.path.join(data_dir, base + ".txt")
            json_path = os.path.join(data_dir, base + ".json")
            if not os.path.exists(json_path):
                # skip samples without synthetic labels
                continue

            with open(txt_path, encoding="utf-8") as f:
                protocol_text = f.read().strip()
            with open(json_path, encoding="utf-8") as f:
                actions_json = f.read().strip()

            # Training prompt: instruct strict JSON output and include automatable actions context
            input_text = (
                "You are a laboratory automation extractor.\n"
                "Extract all actions from the protocol in strict JSON.\n"
                "Rules:\n"
                "- Normalize verbs (e.g., 'incubate').\n"
                "- Include parameters: container, reagent, volume, temperature, duration, speed, other.\n"
                "- 'automatable' true if in the allowed set.\n"
                "- 'confidence' between 0 and 1.\n"
                "- Respond ONLY with a valid JSON array, no extra text.\n"
                f"Automatable actions: {', '.join(self.automatable_actions)}\n\n"
                f"Protocol:\n{protocol_text}\n"
            )

            self.examples.append((input_text, actions_json))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        src, tgt = self.examples[idx]
        inputs = self.tokenizer(
            src,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = self.tokenizer(
            tgt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs["labels"] = labels["input_ids"]
        return {k: v.squeeze(0) for k, v in inputs.items()}
