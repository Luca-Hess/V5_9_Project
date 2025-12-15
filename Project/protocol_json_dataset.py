import os
import json
from typing import Dict, List
from torch.utils.data import Dataset

class ProtocolJsonDataset(Dataset):
    """Paired dataset of protocol text (. txt) and extracted actions (.json)."""
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

        skipped = 0
        invalid_json = []

        for fname in os.listdir(data_dir):
            if not fname.endswith(".txt"):
                continue
            base = fname[:-4]
            txt_path = os.path.join(data_dir, base + ".txt")
            json_path = os.path.join(data_dir, base + ".json")
            
            if not os.path.exists(json_path):
                skipped += 1
                continue

            try:
                with open(txt_path, encoding="utf-8") as f:
                    protocol_text = f.read().strip()
                
                with open(json_path, encoding="utf-8") as f:
                    actions_data = json.load(f)
                
                # Validate JSON structure
                if not self._validate_json_structure(actions_data):
                    invalid_json.append(fname)
                    skipped += 1
                    continue
                
                # Extract actions from different formats
                if isinstance(actions_data, list):
                    if len(actions_data) > 0 and isinstance(actions_data[0], dict) and 'actions' in actions_data[0]:
                        actions = actions_data[0]['actions']
                    else:
                        actions = actions_data
                elif isinstance(actions_data, dict) and 'actions' in actions_data:
                    actions = actions_data['actions']
                else:
                    invalid_json.append(fname)
                    skipped += 1
                    continue
                
                # Serialize to JSON string
                actions_json = json.dumps(actions, ensure_ascii=False)
                input_text = (
                    "Extract actions as JSON:\n"
                    f"{protocol_text}\n"
                )

                # Use simplified prompt - same as inference! (expanded with FLAN-T5 as it is suited for this kind of prompt)
                # input_text = (
                #     "You are a laboratory protocol analyzer. Your task is to extract structured actions from laboratory protocols.\n\n"
                #     "Read the following laboratory protocol carefully and extract each distinct action as a JSON object.\n"
                #     "Each action should include:\n"
                #     "- action: the normalized verb (e.g., 'pipette', 'incubate', 'centrifuge')\n"
                #     "- parameters: an object with container, reagent, volume, temperature, duration, speed, and other fields\n"
                #     "- automatable: boolean indicating if this action can be automated\n"
                #     "- confidence: a number between 0 and 1 indicating your confidence\n\n"
                #     "Return ONLY a valid JSON array of action objects. Do not include any explanatory text.\n\n"
                #     f"Protocol:\n{protocol_text}\n\n"
                #     "JSON output:"
                # )

                input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
                target_tokens = tokenizer.encode(actions_json, add_special_tokens=True)

                if len(input_tokens) > max_len:
                    print(f"Skipping {base}.txt: input {len(input_tokens)} tokens > {max_len}")
                    continue
                    
                if len(target_tokens) > max_len:
                    print(f"Skipping {base}.txt: target {len(target_tokens)} tokens > {max_len}")
                    continue

                self.examples.append((input_text, actions_json))
                
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                skipped += 1
                continue

        print(f"Loaded {len(self.examples)} examples from {data_dir}")
        if skipped > 0:
            print(f"Skipped {skipped} files (missing JSON or invalid format)")
        if invalid_json: 
            print(f"Invalid JSON structure in: {invalid_json[:5]}")


    def _validate_json_structure(self, data) -> bool:
        """Validate that JSON matches expected structure"""
        try:
            # Handle different JSON formats
            if isinstance(data, list):
                # Format 1: [{"protocol": "...", "actions": [...]}]
                if len(data) > 0 and isinstance(data[0], dict) and 'actions' in data[0]:
                    actions = data[0]['actions']
                # Format 2: Direct list of actions
                else:
                    actions = data
            elif isinstance(data, dict) and 'actions' in data:
                # Format 3: {"actions": [...]}
                actions = data['actions']
            else:
                return False
            
            # Check that it's a list of action objects
            if not isinstance(actions, list):
                return False
            
            for action in actions:
                if not isinstance(action, dict):
                    return False
                # Should have at least an 'action' field
                if 'action' not in action and 'name' not in action:
                    return False
            
            return True
        except: 
            return False

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
