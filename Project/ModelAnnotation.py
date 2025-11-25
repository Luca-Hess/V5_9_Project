import os
import json
import torch
import tqdm
import re

from typing import List, Dict, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from peft import LoraConfig, get_peft_model

from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead


# Define automation machine capabilities
class LabAutomationMachine:
    """Defines the capabilities of a fictional lab automation machine"""

    AUTOMATED_ACTIONS = {
        'pipette': {'params': ['source', 'destination', 'volume']},
        'add': {'params': ['reagent', 'container', 'volume']},
        'mix': {'params': ['container', 'duration', 'speed']},
        'incubate': {'params': ['container', 'temperature', 'duration']},
        'centrifuge': {'params': ['container', 'speed', 'duration', 'temperature']},
        'heat': {'params': ['container', 'temperature', 'duration']},
        'cool': {'params': ['container', 'temperature', 'duration']},
        'dispense': {'params': ['reagent', 'destination', 'volume']},
        'aspirate': {'params': ['source', 'volume']},
        'shake': {'params': ['container', 'duration', 'speed']},
        'wait': {'params': ['duration']},
    }

    @classmethod
    def get_action_type(cls, action: str) -> str:
        """Determine if action is automated or manual"""
        action_lower = action.lower()
        if action_lower in cls.AUTOMATED_ACTIONS:
            return 'automated'
        return 'manual' # Default to manual if unknown actions


class AnnotationToScriptConverter:
    """Converts annotation format to automation script format"""

    def __init__(self):
        self.machine = LabAutomationMachine()

    def parse_annotation(self, ann_text: str) -> List[Dict]:
        """Parse .ann format into structured actions"""
        actions = []
        lines = ann_text.strip().split('\n')

        entities = {}
        events = {}
        relations = {}

        # Parse entities
        for line in lines:
            if line.startswith('T'):
                parts = line.split('\t')
                entity_id = parts[0]
                type_span = parts[1].split(' ', 1)
                entity_type = type_span[0]
                text = parts[2] if len(parts) > 2 else ""
                entities[entity_id] = {'type': entity_type, 'text': text}

            elif line.startswith('E'):
                parts = line.split('\t')
                event_id = parts[0]
                event_data = parts[1].split(' ')
                events[event_id] = {'data': event_data}

            elif line.startswith('R'):
                parts = line.split('\t')
                rel_data = parts[1].split(' ')
                relations[parts[0]] = rel_data

        return self._build_action_sequence(entities, events, relations)

    def _build_action_sequence(self, entities, events, relations):
        """Build structured action sequence from parsed annotations"""
        actions = []

        for event_id, event in events.items():
            action_info = {'type': 'unknown', 'params': {}}

            for item in event['data']:
                if ':' in item:
                    key, val = item.split(':', 1)
                    if key == 'Action' and val in entities:
                        action_info['action'] = entities[val]['text']
                        action_info['type'] = self.machine.get_action_type(entities[val]['text'])
                    elif val in entities:
                        action_info['params'][key.lower()] = entities[val]['text']

            # Extract parameters from relations
            for rel_id, rel_data in relations.items():
                if len(rel_data) >= 3:
                    rel_type = rel_data[0]
                    if 'Arg1:' + event_id in ' '.join(rel_data):
                        for part in rel_data[1:]:
                            if 'Arg2:' in part:
                                entity_ref = part.split(':')[1]
                                if entity_ref in entities:
                                    action_info['params'][rel_type.lower()] = entities[entity_ref]['text']

            actions.append(action_info)

        return actions

    def generate_script(self, actions: List[Dict]) -> str:
        """Generate automation script from actions"""
        script_lines = ["# Lab Automation Script", ""]
        current_mode = None

        for i, action in enumerate(actions):
            action_type = action.get('type', 'manual')
            action_name = action.get('action', 'unknown')
            params = action.get('params', {})

            # Add section headers when switching between manual/automated
            if action_type != current_mode:
                if action_type == 'automated':
                    script_lines.append("\n### AUTOMATED SECTION ###")
                else:
                    script_lines.append("\n### MANUAL SECTION ###")
                current_mode = action_type

            # Format action with parameters
            if action_type == 'automated' and action_name.lower() in self.machine.AUTOMATED_ACTIONS:
                param_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
                script_lines.append(f"{action_name.lower()}({param_str})")
            else:
                # Manual action - provide instruction
                param_desc = ", ".join([f"{k}: {v}" for k, v in params.items()])
                script_lines.append(f"# MANUAL: {action_name} ({param_desc})")

        return "\n".join(script_lines)

    def convert(self, ann_text: str) -> str:
        """Full conversion pipeline"""
        actions = self.parse_annotation(ann_text)
        return self.generate_script(actions)

class RewardCalculator:
    """Calculates rewards for RL training based on annotation quality"""

    def __init__(self):
        self.converter = AnnotationToScriptConverter()

    def check_syntax(self, annotation: str) -> float:
        """Check if annotation follows proper .ann syntax"""
        score = 0.0
        lines = annotation.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue
            # Valid lines start with T, E, or R
            if line[0] in ['T', 'E', 'R']:
                score += 1.0

            # Check tab separation
            if '\t' in line:
                score += 0.5

        return min(score / max(len(lines), 1), 1.0)

    def check_action_order(self, annotation: str) -> float:
        """Check if actions follow logical order"""
        try:
            actions = self.converter.parse_annotation(annotation)
            score = 0.0

            # Check that events exist
            if len(actions) > 0:
                score += 0.5

            # Check for reasonable number of actions (not too few or many)
            if 1 < len(actions) < 100:
                score += 0.5

            return score
        except:
            return 0.0


    def check_parameters(self, annotation: str) -> float:
        """Check parameter completeness and validity"""
        try:
            actions = self.converter.parse_annotation(annotation)
            score = 0.0
            total_actions = len(actions)

            if total_actions == 0:
                return 0.0

            for action in actions:
                # Check if action has parameters
                if action.get('params'):
                    score += 0.5

                # Check if action type is identified
                if action.get('action'):
                    score += 0.5

            return score / total_actions
        except:
            return 0.0


    def calculate_reward(self, generated_annotation: str, reference_annotation: str = None) -> float:
        """Calculate total reward score"""
        syntax_score = self.check_syntax(generated_annotation)
        order_score = self.check_action_order(generated_annotation)
        param_score = self.check_parameters(generated_annotation)

        # Weighted combination
        total_reward = (syntax_score * 10.0 +
                        order_score * 5.0 +
                        param_score * 5.0)

        # Bonus for similarity to reference if available
        if reference_annotation:
            similarity = self._calculate_similarity(generated_annotation, reference_annotation)
            total_reward += similarity * 10.0

        return total_reward


    def _calculate_similarity(self, gen: str, ref: str) -> float:
        """Simple token-based similarity"""
        gen_tokens = set(gen.lower().split())
        ref_tokens = set(ref.lower().split())

        if not ref_tokens:
            return 0.0

        intersection = len(gen_tokens & ref_tokens)
        union = len(gen_tokens | ref_tokens)

        return intersection / union if union > 0 else 0.0


class ProtocolDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for fname in os.listdir(data_dir):
            if fname.endswith(".txt"):
                base = fname[:-4]
                txt_path = os.path.join(data_dir, base + ".txt")
                ann_path = os.path.join(data_dir, base + ".ann")

                if not os.path.exists(ann_path):
                    continue

                with open(txt_path, encoding="utf-8") as f:
                    text = f.read().strip()
                with open(ann_path, encoding="utf-8") as f:
                    ann = f.read().strip()

                self.examples.append((text, ann))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, target = self.examples[idx]

        # Add task prefix for T5
        input_text = f"translate protocol to script: {text}"

        inputs = self.tokenizer(input_text,
                                max_length=self.max_len,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt")
        labels = self.tokenizer(target,
                                max_length=self.max_len,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt")
        inputs["labels"] = labels["input_ids"]
        return {key: val.squeeze(0) for key, val in inputs.items()}

tokenizer = T5Tokenizer.from_pretrained("t5-base")

if torch.cuda.is_available():
    base_model = T5ForConditionalGeneration.from_pretrained("t5-large")
    scaler = GradScaler()
    use_amp = True
else:
    base_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    use_amp = False

# Configure LoRA
lora_config = LoraConfig(
    r=64,                # rank
    lora_alpha=128,      # scaling
    target_modules=["q", "k", "v", "o", "wi", "wo"],  # all attention projection layers
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(base_model, lora_config)


train_path = "WLP-Dataset-master/train"
test_path = "WLP-Dataset-master/test"

train_dataset = ProtocolDataset(train_path, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = ProtocolDataset(test_path, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

reward_calculator = RewardCalculator()
converter = AnnotationToScriptConverter()


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(data_loader)

def run_inference_on_specific_protocol(protocol_name: str, converter):
    """Generate automation workflow for a specific test protocol by name"""
    test_path = "WLP-Dataset-master/test"

    # Find the specific protocol in the test dataset
    if protocol_name.endswith(".ann"):
        base_name = protocol_name[:-4]
    else:
        base_name = protocol_name

    protocol_text = base_name + ".txt"
    reference_annotation = base_name + ".ann"

    txt_path = os.path.join(test_path, protocol_text)
    ann_path = os.path.join(test_path, reference_annotation)

    with open(txt_path, encoding="utf-8") as f:
        text = f.read().strip()
    with open(ann_path, encoding="utf-8") as f:
        ann = f.read().strip()

    print(f"\n{'=' * 80}")
    print(f"PROTOCOL: {protocol_name}")
    print(f"{'=' * 80}")
    print(f"\nOriginal Protocol Text:")
    print(text)
    print(f"\n{'-' * 80}")
    print(f"Reference Annotation (.ann file):")
    print(ann)
    print(f"\n{'-' * 80}")


    # Convert annotation to automation script (rule-based)
    script_from_reference = converter.convert(ann)

    print(f"Automation Script (from reference annotation):")
    print(script_from_reference)
    print(f"\n{'=' * 80}\n")

if __name__ == "__main__":

    # Test on protocol_2 (index 2 in test set)
    print("Testing automation workflow generation on test data (BEFORE training):")
    run_inference_on_specific_protocol("protocol_224", converter)

    # Supervised pre-training
    print("Starting supervised pre-training...")
    num_pretraining_epochs = 3

    for epoch in range(num_pretraining_epochs):
        model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Pretrain Epoch {epoch + 1}")

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

        test_loss = evaluate(model, test_loader, device)
        scheduler.step(test_loss)
        print(f"Pretrain Epoch {epoch + 1} finished, Test Loss: {test_loss:.4f}")

    # Reinforcement Learning fine-tuning
    print("Starting RL fine-tuning...")

    # Wrap model for PPO
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model)
    ppo_model.to(device)

    ppo_config = PPOConfig(
        learning_rate = 1e-5,
        batch_size = 4,
        mini_batch_size = 4,
        gradient_accumulation_steps = 1
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        tokenizer=tokenizer
    )

    # Training Loop
    num_rl_epochs = 2
    for epoch in range(num_rl_epochs):
        loop = tqdm.tqdm(train_loader, desc=f"RL Epoch {epoch + 1}")

        for batch in loop:
            protocol_texts = [tokenizer.decode(ids, skip_special_tokens=True)
                              for ids in batch['input_ids']]
            reference_annotations = [tokenizer.decode(ids, skip_special_tokens=True)
                                     for ids in batch['labels']]

            # Generate annotations
            query_tensors = batch['input_ids'].to(device)
            response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=512)

            # Decode generated annotations
            generated_annotations = [tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                                     for r in response_tensors]

            # Calculate rewards
            rewards = []
            for gen_ann, ref_ann in zip(generated_annotations, reference_annotations):
                reward = reward_calculator.calculate_reward(gen_ann, ref_ann)
                rewards.append(torch.tensor(reward, device=device))

            # PPO update
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            loop.set_postfix(mean_reward=torch.tensor(rewards).mean().item())

        print(f"RL Epoch {epoch + 1} finished, Mean Reward: {torch.tensor(rewards).mean().item():.4f}")

    # Save final model
    ppo_model.save_pretrained("protocol_annotation_rl_model")
    tokenizer.save_pretrained("protocol_annotation_rl_model")


def generate_automation_workflow(protocol_text: str, model, tokenizer, device):
    """Two-stage generation: protocol → annotation → automation script"""
    model.eval()

    # Stage 1: Generate annotation from protocol
    input_text = f"translate protocol to automation: {protocol_text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    annotation_output = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )

    annotation = tokenizer.decode(annotation_output[0], skip_special_tokens=True)

    # Stage 2: Convert annotation to automation script (rule-based)
    script = converter.convert(annotation)
    return annotation, script

if __name__ == "__main__":

    # Example usage
    example_protocol = """Standard RNA Synthesis (E2050)
    Thaw the necessary kit components.
    Mix and pulse-spin in microfuge to collect solutions to the bottoms of tubes.
    Keep on ice.
    Incubate at 37°C for 2 hours."""

    print("\n" + "="*60)
    print("EXAMPLE AUTOMATION SCRIPT:")
    print("="*60)
    annotation, script = generate_automation_workflow(example_protocol, model, tokenizer, device)
    print("\nGenerated Annotation:")
    print(annotation)
    print("\nGenerated Automation Script:")
    print(script)



# # Protocol 103 annotation example
# print(annotate_protocol("""Isolation Of Total DNA From NC64A Chlorella
# Inoculate 500 mL flasks with NC64A chlorella, each flask to contain 360 mL of cells at 1.2 X 106 cells/mL in MBBM.
# Incubate the flasks at 25°C for 72 hours, with continuous light and shaking.
# Count the cells.
# Concentrate aliquots of NC64A chlorella, each aliquot to contain 6.0 X 109 cells.
# If the cells are to be infected with virus, infection should be at an moi (multiplicity of infection) of 3-5.
# Centrifuge the samples in the Sorvall GSA rotor at 5,000 rpm, 5 min, 4°C to harvest the cells.
# Wash the cells 1X with sterile d-H2O in the Sorvall HB-4 rotor at 5,000 rpm, 5 min, 4°C.
# Quick freeze the cells pellets with liquid N2 and store the frozen pellets at -80°C until ready for processing.
# Resuspend each frozen sample with 5.0 mL of 50 mM NaHPO4, pH 7.4, 2.0 M NaCl.
# Heat the samples at 65°C for 30 min (leave in the MSK flasks during the heating).
# Break the samples a second time in the MSK for 30 sec with CO2 cooling.
# Recover the homogenates to clean tubes (SS34 plastic tubes).
# Remove a 0.3 mL aliquot from each sample to microfuge tubes for determination of the 			original DNA concentration for each sample (use the fluorometric procedure).
# Treat each sample with 500 µL of proteinase K for 60 min at 37°C (add 200 µL/sample).
# Heat the samples at 65°C for 5 min.
# Add 20% SDS to each sample to a final concentration of 1% (add 500 µL/sample).
# Add 2.7 mL of 5 M KOAc to each sample, mix well (a final concentration of 1 M).
# Incubate the samples in the cold room for 30 min.
# Treat the samples with RNAse at 200 µg/mL for 2 hours at 37°C (add 200 µL/sample).
# Precipitate the DNAs in the samples by adding 2X volumes of 100% EtOH (approximately 30 mL/sample).
# Centrifuge the tubes in the Sorvall HB-4 rotor at 10,000 rpm, 15 min, 4°C.
# Resuspend each DNA sample with 3.5 mL of 50 mM Tris-HCl, pH 8.0, 10 mM EDTA.
# Dialyze the samples overnight at 4°C against several changes of 50 mM Tris-HCl, pH 8.0, 10 mM EDTA.
# Add 375 µL of 3 M NaOAc to each sample.
# Centrifuge the samples in the Sorvall SS34 rotor at 10,000 rpm, 20 min, 4°C.
# Wash the DNA pellets 1X with 10 mL of 70% EtOH in the Sorvall SS34 rotor at 10,000 rpm, 5 min, 4°C.
# Dry the pellets briefly (10-15 min) in the vacuum desiccator to remove the EtOH.
# Determine the DNA concentration of the samples using the fluorometric procedure.
# Run CsCl gradients.
# Centrifuge the required volume of cells for each aliquot in the Sorvall GSA rotor at 5,000 rpm, 5 min, 4°C.
# Resuspend each aliquot with 160 mL of fresh MBBM.
# Incubate the flasks for 45-60 min at 25°C for the cells to acclimate to the new media.
# Incubate the flasks for the desired length of time at 25°C with continuous light and shaking.
# Discard the supernatants.
# Break the cells in the MSK mechanical homogenizer with 5.0 gm of 0.3 mm glass beads for 60 sec, with CO2 cooling.
# Wash the glass beads with 50 mM NaHPO4, pH 7.4, 2.0 M NaCl.
# Add the washes to the homogenates.
# Centrifuge the sample in the Sorvall SS34 rotor at 14,000 rpm, 20 min, 4°C.
# Decant the supernatants to clean tubes.
# Store at -20°C for 2-3 hours.
# Discard the supernatants.
# Precipitate the DNAs by adding an equal volume of isopropanol to each tube.
# Mix well and hold at room temperature for 30 min.
# Discard the supernatants.
# Resuspend the pellets with 2.0 mL of 1X TE buffer."""))
