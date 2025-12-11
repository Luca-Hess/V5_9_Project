import os
import json
from functools import total_ordering

import torch
import tqdm
import re

from typing import List, Dict, Optional

from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from peft import LoraConfig, get_peft_model

from LLMExtraction import ImprovedAnnotationToScriptConverter #rules based
from protocol_json_dataset import ProtocolJsonDataset
from json_script_converter import JsonScriptConverter


def load_model(use_cuda: bool = torch.cuda.is_available()) -> (T5ForConditionalGeneration, T5Tokenizer, torch.device):
    """Load T5 model and tokenizer"""
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    base = T5ForConditionalGeneration.from_pretrained('t5-base')

    lora = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(base, lora)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    return model, tokenizer, device

def train_json_extractor(
        data_dir: str,
        output_dir: str = 'json_extractor_model',
        batch_size: int = 8,
        epochs: int = 10,
        lr: float = 3e-4,
        max_len: int = 1024,
        automatable_actions: Optional[List[str]] = None,
        val_split: float = 0.2
):
    """Train T5 model to extract JSON actions from protocols"""
    model, tokenizer, device = load_model()
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    # Load full dataset
    full_dataset = ProtocolJsonDataset(
        data_dir, tokenizer, max_len, automatable_actions
    )
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    def eval_json_rate(samples: List[Dict[str, str]]) -> float:
        ok = 0
        for s in samples:
            try:
                json.loads(s['pred'])
                ok += 1
            except Exception:
                pass
        return ok / max(len(samples), 1)

    def validate(val_loader_iter, max_samples: int = 32) -> (float, float):
        model.eval()
        total_loss = 0.0
        samples = []
        count = 0

        with torch.no_grad():
            for batch in val_loader_iter:
                if count >= max_samples:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                total_loss += out.loss.item()
                count += 1

                # Generate predictions for parse rate
                pred_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=max_len,
                    num_beams=4,
                    early_stopping=True
                )
                for i, pred in enumerate(pred_ids):
                    pred_text = tokenizer.decode(pred, skip_special_tokens=True).strip()
                    tgt = tokenizer.decode(
                        batch['labels'][i],
                        skip_special_tokens=True
                    ).strip()
                    samples.append({"pred": pred_text, "tgt": tgt})

        avg_loss = total_loss / count if count > 0 else 0.0
        parse_rate = eval_json_rate(samples)
        return avg_loss, parse_rate

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


    for epoch in range(epochs):
        model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp:
                with autocast():
                    out = model(**batch)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

        # Validation
        val_loss, val_rate = validate(val_loader, max_samples=32)
        print(
            f"Epoch {epoch+1} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Val JSON Parse Rate: {val_rate:.2%}"
        )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def _ensure_valid_json(text: str) -> Optional[List[Dict]]:
    """Attempt to fix common JSON issues in model output"""
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        s = text.strip()
        start = s.find('[')
        end = s.rfind(']') + 1
        if start >= 0 and end > start:
            return json.loads(s[start:end])

    except Exception:
        return None
    return None

def infer_actions_from_protocol(
        protocol_text: str,
        model_dir: str = 'json_extractor_model',
        automatable_actions: Optional[List[str]] = None,
        max_len: int = 1024
) -> List[Dict]:
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        automatable_actions = automatable_actions or [
            "pipette","dispense","aspirate","mix","shake",
            "incubate","heat","cool","centrifuge","wait"
        ]

        prompt = (
            "You are a laboratory automation extractor.\n"
            "Extract all actions from the protocol in strict JSON.\n"
            "Rules:\n"
            "- Normalize verbs (e.g., 'incubate').\n"
            "- Include parameters: container, reagent, volume, temperature, duration, speed, other.\n"
            "- 'automatable' true if in the allowed set.\n"
            "- 'confidence' between 0 and 1.\n"
            "- Respond ONLY with a valid JSON array, no extra text.\n"
            f"Automatable actions: {', '.join(automatable_actions)}\n\n"
            f"Protocol:\n{protocol_text}\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_len, truncation=True).to(device)
        pred_ids = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )

        text = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

        # Validate JSON, retry with greedy if needed
        actions = _ensure_valid_json(text)
        if actions is None:
            pred_ids = model.generate(
                **inputs,
                max_length=max_len,
                do_sample=False,
                num_beams=1
            )
            text = tokenizer.decode(pred_ids[0],
                                    skip_special_tokens=True).strip()
            actions = _ensure_valid_json(text)
            if actions is None:
                raise ValueError("Failed to parse JSON from model output.")

        return actions

def save_actions_json_from_file(txt_path: str,
                                output_path: str,
                                model_dir: str = 'json_extractor_model',
                                automatable_actions: Optional[List[str]] = None):
    """Load protocol from .txt file, infer actions, and save to .json file"""
    with open(txt_path, 'r', encoding="utf-8") as f:
        protocol_text = f.read().strip()
    actions = infer_actions_from_protocol(
        protocol_text,
        model_dir=model_dir,
        automatable_actions=automatable_actions
    )
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(actions, f, indent=2)
    print(f"Saved actions JSON to {output_path}")


def evaluate_test_set(
        test_dir: str,
        model_dir: str = 'json_extractor_model',
        output_dir: str = 'test_predictions',
        automatable_actions: Optional[List[str]] = None,
        max_len: int = 1024
):
    """
    Evaluate model on test set with ground truth JSON files

    For each protocol_X.txt in test_dir:
    1. Generate predicted actions JSON
    2. Save to output_dir/protocol_X.pred.json
    3. If protocol_X.json exists, compute metrics
    """
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}\n")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    automatable_actions = automatable_actions or [
        "pipette","dispense","aspirate","mix","shake",
        "incubate","heat","cool","centrifuge","wait"
    ]

    # Metrics tracking
    total_files = 0
    successful_parses = 0
    failed_parses = []

    # find all .txt files in test_dir
    txt_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]

    print(f"Found {len(txt_files)} protocols.")

    for fname in tqdm.tqdm(txt_files, desc='Processing test set'):
        total_files += 1
        base = fname[:-4]
        txt_path = os.path.join(test_dir, fname)
        pred_path = os.path.join(output_dir, f'{base}.pred.json')
        gt_path = os.path.join(test_dir, f'{base}.json')

        # Load protocol text
        with open(txt_path, 'r', encoding="utf-8") as f:
            protocol_text = f.read().strip()

        # Generate predicted actions
        try:
            actions = infer_actions_from_protocol(
                protocol_text,
                model_dir=model_dir,
                automatable_actions=automatable_actions,
                max_len=max_len
            )

            # Save predicted actions
            with open(pred_path, 'w', encoding="utf-8") as f:
                json.dump(actions, f, indent=2)

            successful_parses += 1

        except Exception as e:
            print(f'Failed to parse {fname}: {e}')
            failed_parses.append(fname)
            continue

    # Print summary
    print(f'\n{"="*60}')
    print("Test Set Evaluation Summary")
    print(f'{"="*60}\n')
    print(f'Total Protocols Processed:  {total_files}')
    print(f'Successful JSON Parses:     {successful_parses}')
    print(f'Failed JSON Parses:         {len(failed_parses)}')
    print(f'JSON Parse rate:            {successful_parses / total_files:.2%}')

    if failed_parses:
        print("\nFailed Parses:")
        for fname in failed_parses:
            print(f' - {fname}')

    print(f'\nPredicted JSON files saved to: {output_dir}\n')
    print(f'{"="*60}\n')

    return {
        'total': total_files,
        'successful': successful_parses,
        'failed': len(failed_parses),
        'parse_rate': successful_parses / total_files if total_files > 0 else 0.0
    }


if __name__ == "__main__":

    # Phase 1: Training
    print("\n" + "="*60)
    print("Phase 1: Training JSON Extraction Model")
    print("="*60 + "\n")

    train_json_extractor(
        data_dir='WLP-Dataset-master/train',
        output_dir='json_extractor_model',
        batch_size=8,
        epochs=10,
        lr=3e-4,
        max_len=1024,
        automatable_actions=[
            "pipette","dispense","aspirate","mix","shake",
            "incubate","heat","cool","centrifuge","wait"
        ],
        val_split=0.2
    )

    # Phase 2: Testing
    print("\n" + "="*60)
    print("Phase 2: Evaluating JSON Extraction Model on Test Set")
    print("="*60 + "\n")

    test_results = evaluate_test_set(
        test_dir='WLP-Dataset-master/test',
        model_dir='json_extractor_model',
        output_dir='test_predictions',
        automatable_actions=[
            "pipette","dispense","aspirate","mix","shake",
            "incubate","heat","cool","centrifuge","wait"
        ],
        max_len=1024
    )

    # Phase 3: Example script generation
    print("\n" + "="*60)
    print("Phase 3: Generating Automation Script from Example Protocol")
    print("="*60 + "\n")

    example_pred_json = 'test_predictions/protocol_224.pred.json'

    if os.path.exists(example_pred_json):
        with open(example_pred_json, 'r', encoding='utf-8') as f:
            actions = json.load(f)

        rb_converter = JsonScriptConverter()
        script = rb_converter.generate_script(actions)

        print("Example Automation Script (protocol_224):\n")
        print(script)
    else:
        print(f"Example predicted JSON file not found: {example_pred_json}")