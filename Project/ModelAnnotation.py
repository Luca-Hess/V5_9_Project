import os
import json
import re
import math

import torch
import tqdm

from typing import List, Dict, Optional, Tuple

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PeftModel

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from json_repair import repair_json


from protocol_json_dataset import ProtocolJsonDataset
from json_script_converter import JsonScriptConverter


# Define expected JSON structure
ACTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "parameters": {
                "type": "object",
                "properties": {
                    "container": {"type": ["string", "null"]},
                    "reagent": {"type": ["string", "null"]},
                    "volume": {"type": ["string", "null"]},
                    "temperature": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "speed": {"type": ["string", "null"]},
                    "other": {"type": "object", "additionalProperties": True}
                }
            },
            "automatable": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["action", "parameters"]
    }
}


def load_model(use_cuda: bool = torch.cuda.is_available()) -> Tuple[T5ForConditionalGeneration, T5Tokenizer, torch.device]:
    """Load Flan-UL2 model and tokenizer"""
    cache_dir = '/cfs/earth/scratch/hessluc1/ADL/cache' # because not enough space otherwise
    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('google/flan-ul2', cache_dir=cache_dir)

    # Added 8-bit quantization due to OOM issues on GPU
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    base = T5ForConditionalGeneration.from_pretrained(
        'google/flan-ul2',
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map={"": 0},  # Force to GPU 0
        low_cpu_mem_usage=True,
        dtype=torch.float16
    )

    # Enable gradient checkpointing for memory efficiency
    base.gradient_checkpointing_enable()

    # Prepare model for k-bit training
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=64, # increased from 128, then decreased to 64 because of large new model size (3B param)
        lora_alpha=128, # decreased from 256
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        lora_dropout=0.03, # decr. from 0.05 less regularization since underfitting
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    device = torch.device("cuda" if use_cuda else "cpu")

    return model, tokenizer, device

def train_json_extractor(
        data_dir: str,
        output_dir: str = 'json_extractor_model',
        batch_size: int = 1, # reduced due to large FLAN-UL2 model
        gradient_accumulation_steps: int = 16,
        epochs: int = 30,
        lr: float = 3e-5,
        max_len: int = 2048, # native context len for FLAN-UL2
        automatable_actions: Optional[List[str]] = None,
        val_split: float = 0.2
):
    """Train T5 model to extract JSON actions from protocols"""
    model, tokenizer, device = load_model()
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    # Load full dataset
    full_dataset = ProtocolJsonDataset(
        data_dir, tokenizer, max_len, automatable_actions
    )
    
    if len(full_dataset) == 0:
        raise ValueError(f"No valid training examples found in {data_dir}.  Check that . txt and .json pairs exist.")
    
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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8) # eps added for stability after FLAN training produced NaN loss

    # Total training steps
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.05 * total_steps) # 10% warmup, reduced to 5%

    # Gradient clipping for stability
    max_grad_norm = 1.0 

    # Linear warmup followed by cosine annealing
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    

    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    best_val_loss = float('inf')
    patience = 5 # increased from 5, loss got stuck
    patience_counter = 0

    def eval_json_rate(samples: List[Dict[str, str]]) -> float:
        ok = 0
        for s in samples:
            result = _ensure_valid_json(s['pred'])
            if result is not None:
                ok += 1
        return ok / max(len(samples), 1)

    def validate(val_loader_iter, max_samples: int = 32) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        samples = []
        valid_count = 0
        count = 0

        with torch.no_grad():
            for batch in val_loader_iter:
                if count >= max_samples:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}

                # Clear cache before validation
                torch.cuda.empty_cache()

                try:
                    # Forward pass for loss
                    out = model(**batch)
                    loss = out.loss

                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        valid_count += 1

                    # Generate predictions for parse rate
                    pred_ids = model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=256, # REDUCED from 512, caused OOM
                        num_beams=1, # REDUCED from 4, caused OOM ,
                        do_sample=False,
                        early_stopping=True
                    )

                    for i, pred in enumerate(pred_ids):
                        pred_text = tokenizer.decode(pred, skip_special_tokens=True).strip()
                        tgt = tokenizer.decode(
                            batch['labels'][i],
                            skip_special_tokens=True
                        ).strip()
                        samples.append({"pred": pred_text, "tgt": tgt})

                    count += 1

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print("Warning: Out of memory during validation. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            avg_loss = total_loss / valid_count if valid_count > 0 else float('nan')
            valid_rate = valid_count / count if count > 0 else 0.0

            return avg_loss, valid_rate

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model.train()
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(loop):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(**batch)
                    loss = out.loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                out = model(**batch)
                loss = out.loss / gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # Step scheduler after each optimizer step
                global_step += 1

            loop.set_postfix(loss=loss.item() * gradient_accumulation_steps)

        # Validation
        model.eval()
        val_loss, val_rate = validate(iter(val_loader), max_samples=16)
        model.train()

        print(
            f"Epoch {epoch+1} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Val JSON Parse Rate: {val_rate:.2%} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  Saved best model to {output_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Model saved to {output_dir}")


def _ensure_valid_json(text: str, verbose: bool = False) -> Optional[List[Dict]]:
    """Attempt multiple strategies to extract/repair JSON from model output"""
    
    # Direct parsing
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
    except Exception as e:
        if verbose:
            print(f"Direct parse failed: {e}")
    
    # Extract JSON array from surrounding text
    try:
        text_stripped = text.strip()
        start = text_stripped.find('[')
        end = text_stripped.rfind(']') + 1
        if start >= 0 and end > start: 
            json_str = text_stripped[start:end]
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
    except Exception as e:
        if verbose:
            print(f"Array extraction failed: {e}")
    
    # Use json_repair library
    try:
        repaired = repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
    except Exception as e:
        if verbose:
            print(f"json_repair failed: {e}")
    
    # Fix common issues manually
    try:
        fixed = text.strip()
        # Remove markdown code blocks
        fixed = re.sub(r'^```(?:json)?\s*', '', fixed)
        fixed = re.sub(r'\s*```$', '', fixed)
        # Fix trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        # Ensure it starts with [ and ends with ]
        if not fixed.startswith('['):
            fixed = '[' + fixed
        if not fixed.endswith(']'):
            fixed = fixed + ']'
        
        result = json.loads(fixed)
        if isinstance(result, list):
            return result
    except Exception as e:
        if verbose:
            print(f"Manual fixes failed: {e}")
    
    # Extract individual objects
    try:
        # Find all {... } patterns
        objects = re.findall(r'\{[^{}]*\}', text)
        if objects:
            parsed = []
            for obj in objects:
                try:
                    parsed_obj = json.loads(obj)
                    parsed.append(parsed_obj)
                except: 
                    continue
            if parsed:
                return parsed
    except Exception as e:
        if verbose:
            print(f"Object extraction failed: {e}")
    
    # Nested object extraction (handles nested braces)
    try:
        parsed = []
        depth = 0
        current_obj = ""
        in_string = False
        escape = False
        
        for char in text:
            if escape: 
                current_obj += char
                escape = False
                continue
            
            if char == '\\':
                escape = True
                current_obj += char
                continue
            
            if char == '"' and not escape:
                in_string = not in_string
                current_obj += char
                continue
            
            if not in_string:
                if char == '{':
                    if depth == 0:
                        current_obj = "{"
                    else:
                        current_obj += char
                    depth += 1
                elif char == '}': 
                    depth -= 1
                    current_obj += char
                    if depth == 0 and current_obj: 
                        try:
                            obj = json.loads(current_obj)
                            parsed.append(obj)
                            current_obj = ""
                        except: 
                            current_obj = ""
                else:
                    if depth > 0:
                        current_obj += char
            else:
                current_obj += char
        
        if parsed:
            return parsed
    except Exception as e:
        if verbose:
            print(f"Nested extraction failed: {e}")
    
    if verbose:
        print(f"All parsing strategies failed. Raw output (first 300 chars):\n{text[:300]}")
    
    return None

def load_trained_model(model_dir: str = 'json_extractor_model'):
    """Load a trained PEFT model for inference"""
    cache_dir = '/cfs/earth/scratch/hessluc1/ADL/cache'
    
    print(f"Loading model from {model_dir}")
    
    # Load PEFT config
    try:
        config = PeftConfig.from_pretrained(model_dir)
        base_model_path = config.base_model_name_or_path
        print(f"Base model: {base_model_path}")
    except Exception as e:
        print(f"Warning: Could not load PEFT config: {e}")
        base_model_path = "google/flan-ul2"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir)
    
    # 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # Load base model
    base_model = T5ForConditionalGeneration.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    
    # Load PEFT adapters
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Model loaded successfully!\n")
    return model, tokenizer, device

def infer_actions_from_protocol(
        protocol_text: str,
        model, 
        tokenizer,
        device, 
        automatable_actions: Optional[List[str]] = None,
        max_len: int = 1024,
        use_constrained_decoding: bool = True,
        num_fallback_attempts: int = 2
) -> List[Dict]:
    """
    Infer actions from protocol text using Flan-T5 with optional constrained decoding
    
    Args:
        protocol_text: Input protocol text
        model: Pretrained FLAN-UL2 model
        tokenizer: Corresponding tokenizer
        device: Torch device (CPU/GPU)
        automatable_actions: List of automatable action names
        max_len: Maximum sequence length
        use_constrained_decoding: Whether to use schema-constrained generation
        num_fallback_attempts: Number of fallback attempts if constrained decoding fails
    
    Returns:
        List of action dictionaries
    """

    automatable_actions = automatable_actions or [
        "pipette", "dispense", "aspirate", "mix", "shake",
        "incubate", "heat", "cool", "centrifuge", "wait"
    ]

    # Use the same prompt as training. Shortened again because FLAN T5 can only handle 512 tokens!
    prompt = (
        "Extract actions as JSON:\n"
        f"{protocol_text}\n"
    )
    # prompt = (
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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_len,
        truncation=True
    ).to(device)

    # Try constrained decoding first
    if use_constrained_decoding:
        print("Using constrained decoding with JSON schema...")
        try:
            parser = JsonSchemaParser(ACTION_SCHEMA)
            prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=max_len,
                    prefix_allowed_tokens_fn=prefix_fn,
                    num_beams=1,  # Constrained decoding works best with greedy
                    early_stopping=True
                )

            
            text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            print(f"Constrained output (first 200 chars): {text[:200]}")
            
            # Should be valid JSON, but verify
            actions = json.loads(text)
            if isinstance(actions, list):
                print(f"Successfully generated {len(actions)} actions with constrained decoding")
                return actions
            elif isinstance(actions, dict):
                print(f"Successfully generated 1 action with constrained decoding")
                return [actions]
                
        except Exception as e: 
            print(f"Constrained decoding failed: {e}")
            print("Falling back to standard generation...")
    
    # Fallback:  Standard generation with multiple attempts
    generation_configs = [
        # Config 1: Greedy decoding
        {
            'max_length': max_len,
            'num_beams': 1,
            'do_sample': False,
            'early_stopping': True,
        },
        # Config 2: Beam search
        {
            'max_length': max_len,
            'num_beams': 4,
            'do_sample': False,
            'early_stopping': True,
        },
        # Config 3: Sampling with low temperature
        {
            'max_length': max_len,
            'do_sample': True,
            'temperature': 0.3,
            'top_p': 0.9,
        },
    ]

    for attempt, config in enumerate(generation_configs[: num_fallback_attempts], 1):
        print(f"Fallback attempt {attempt}/{num_fallback_attempts}")
        
        with torch.no_grad():
            pred_ids = model.generate(**inputs, **config)

        text = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
        print(f"Raw output (first 200 chars): {text[:200]}")
        
        actions = _ensure_valid_json(text, verbose=(attempt == num_fallback_attempts))

        if actions is not None and len(actions) > 0:
            print(f"Successfully parsed JSON with {len(actions)} actions")
            return actions
        else:
            print(f"Attempt {attempt} failed to produce valid JSON")

    print(f"Warning: All generation attempts failed to produce valid JSON")
    return []


def evaluate_test_set(
        test_dir: str,
        model_dir: str = 'json_extractor_model',
        output_dir: str = 'test_predictions',
        automatable_actions: Optional[List[str]] = None,
        max_len: int = 2048
):
    """
    Evaluate model on test set with ground truth JSON files
    """
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}\n")

    model, tokenizer, device = load_trained_model(model_dir)

    os.makedirs(output_dir, exist_ok=True)

    automatable_actions = automatable_actions or [
        "pipette","dispense","aspirate","mix","shake",
        "incubate","heat","cool","centrifuge","wait"
    ]

    total_files = 0
    successful_parses = 0
    failed_parses = []

    action_name_matches = 0
    total_gt_actions = 0
    total_pred_actions = 0
    parameter_tp = 0
    parameter_fp = 0
    parameter_fn = 0

    txt_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]

    print(f"Found {len(txt_files)} protocols.")

    for fname in tqdm.tqdm(txt_files, desc='Processing test set'):
        total_files += 1
        base = fname[:-4]
        txt_path = os.path.join(test_dir, fname)
        pred_path = os.path.join(output_dir, f'{base}.pred.json')
        gt_path = os.path.join(test_dir, f'{base}.json')

        with open(txt_path, 'r', encoding="utf-8") as f:
            protocol_text = f.read().strip()
        
        # Check token length before processing
        input_text = f"Extract actions as JSON:\n{protocol_text}\n"
        input_tokens = tokenizer.encode(input_text, add_special_tokens=True)

        if len(input_tokens) > max_len:
            print(f"Skipping {fname}: {len(input_tokens)} tokens > {max_len}")
            failed_parses.append((fname, f"Input too long: {len(input_tokens)} tokens"))
            continue

        try:
            torch.cuda.empty_cache()

            pred_actions = infer_actions_from_protocol(
                protocol_text,
                model=model,
                tokenizer=tokenizer,
                device=device,
                automatable_actions=automatable_actions,
                max_len=max_len,
                use_constrained_decoding=True, # added after initial implementation failed to resolve to JSONs
                num_fallback_attempts=2  # Reduced attempts for speed
            )

            with open(pred_path, 'w', encoding="utf-8") as f:
                json.dump(pred_actions, f, indent=2)

            if len(pred_actions) > 0:
                successful_parses += 1

            if os.path.exists(gt_path):
                with open(gt_path, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)

                if isinstance(gt_data, list) and len(gt_data) > 0:
                    if 'actions' in gt_data[0]:
                        gt_actions = gt_data[0]['actions']
                    else:
                        gt_actions = gt_data
                else:
                    gt_actions = gt_data

                total_gt_actions += len(gt_actions)
                total_pred_actions += len(pred_actions)

                gt_names = [a.get('action', '').lower() for a in gt_actions]
                pred_names = [a.get('action', '').lower() for a in pred_actions]

                for pn in pred_names:
                    if pn in gt_names: 
                        action_name_matches += 1

                for gt_action in gt_actions:
                    gt_name = gt_action.get('action', '').lower()
                    gt_params = gt_action.get('params', {}) or gt_action.get('parameters', {})

                    pred_action = next(
                        (p for p in pred_actions if p.get('action', '').lower() == gt_name),
                        None
                    )

                    if pred_action: 
                        pred_params = pred_action.get('parameters', {}) or pred_action.get('params', {})

                        gt_flat = {k: v for k, v in gt_params.items() if k != 'other' and v is not None}
                        if gt_params.get('other'):
                            gt_flat.update(gt_params['other'])

                        pred_flat = {k: v for k, v in pred_params.items() if k != 'other' and v is not None}
                        if pred_params.get('other'):
                            pred_flat.update(pred_params['other'])

                        gt_keys = set(gt_flat.keys())
                        pred_keys = set(pred_flat.keys())

                        parameter_tp += len(gt_keys & pred_keys)
                        parameter_fp += len(pred_keys - gt_keys)
                        parameter_fn += len(gt_keys - pred_keys)

        except Exception as e:
            print(f'Failed to process {fname}: {e}')
            failed_parses.append((fname, str(e)))
            continue

    parse_rate = successful_parses / total_files if total_files > 0 else 0.0

    action_precision = (action_name_matches / total_pred_actions) if total_pred_actions > 0 else 0.0
    action_recall = (action_name_matches / total_gt_actions) if total_gt_actions > 0 else 0.0
    action_f1 = (2 * action_precision * action_recall / (action_precision + action_recall)
                 if (action_precision + action_recall) > 0 else 0.0)

    param_precision = (parameter_tp / (parameter_tp + parameter_fp) if (parameter_tp + parameter_fp) > 0 else 0.0)
    param_recall = (parameter_tp / (parameter_tp + parameter_fn) if (parameter_tp + parameter_fn) > 0 else 0.0)
    param_f1 = (2 * param_precision * param_recall / (param_precision + param_recall)
                if (param_precision + param_recall) > 0 else 0.0)

    print(f'\n{"="*60}')
    print("Test Set Evaluation Summary")
    print(f'{"="*60}\n')
    print(f'Total Protocols Processed:   {total_files}')
    print(f'Successful JSON Parses:     {successful_parses}')
    print(f'Failed JSON Parses:          {len(failed_parses)}')
    print(f'JSON Parse rate:            {parse_rate:.2%}')

    print(f'\nAction Extraction:')
    print(f'Ground Truth Actions:        {total_gt_actions}')
    print(f'Predicted Actions:          {total_pred_actions}')
    print(f'Correct Action Names:       {action_name_matches}')
    print(f'Action Precision:           {action_precision:.2%}')
    print(f'Action Recall:              {action_recall:.2%}')
    print(f'Action F1 Score:            {action_f1:.2%}')

    print(f'\nParameter Extraction:')
    print(f'True Positives (TP):        {parameter_tp}')
    print(f'False Positives (FP):       {parameter_fp}')
    print(f'False Negatives (FN):       {parameter_fn}')
    print(f'Parameter Precision:        {param_precision:.2%}')
    print(f'Parameter Recall:           {param_recall:.2%}')
    print(f'Parameter F1 Score:         {param_f1:.2%}')

    if failed_parses:
        print("\nFailed Parses:")
        for fname, error in failed_parses[:5]: 
            print(f' - {fname}: {error[:60]}...')
        if len(failed_parses) > 5:
            print(f' ... and {len(failed_parses) - 5} more.')

    print(f'\nPredicted JSON files saved to: {output_dir}\n')
    print(f'{"="*60}\n')

    return {
        'total':  total_files,
        'successful': successful_parses,
        'failed': len(failed_parses),
        'parse_rate': parse_rate,
        'action_precision': action_precision,
        'action_recall': action_recall,
        'action_f1': action_f1,
        'parameter_precision': param_precision,
        'parameter_recall': param_recall,
        'parameter_f1':  param_f1
    }


if __name__ == "__main__": 

    model_dir = 'json_extractor_model'

    model_exists = (
        os.path.exists(model_dir) and
        os.path.exists(os.path.join(model_dir, 'adapter_config.json')) and
        (os.path.exists(os.path.join(model_dir, 'adapter_model.safetensors')) or
         os.path.exists(os.path.join(model_dir, 'adapter_model.bin')))
    )

    if model_exists:
        print("\n" + "="*60)
        print("Existing Model Found")
        print("Model directory:", model_dir)
        print("Skipping training phase.")
        print("="*60 + "\n")

    else:
        print("\n" + "="*60)
        print("Phase 1: Training JSON Extraction Model")
        print("="*60 + "\n")


        train_json_extractor(
            data_dir='WLP-Dataset-master/train',
            output_dir='json_extractor_model',
            batch_size=1,  # reduced from 8 - GPU memory constraint
            gradient_accumulation_steps=16,  # simulated batch size of 16
            epochs=30,  # Increased from 15 - early stopping will prevent overfitting
            lr=5e-5, # decreased from 1e-4 for large model, increased from 3e-5 for faster convergence
            max_len=2048, # native context len for FLAN-UL2, increased from 512 for FLAN T5
            automatable_actions=[
                "pipette","dispense","aspirate","mix","shake",
                "incubate","heat","cool","centrifuge","wait"
            ],
            val_split=0.2
        )

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

    # Learnings and Future Directions:
    # - Switching to UL2 model and trying to directly decode to JSON was a poor decision.
    # - Better approach would likely have been to train a smaller model on txt-to-ann task and then add a second
    # stage in which the extracted actions are parsed somehow with invariance (incubate == incubating == incubated etc.)
    # From this more structured text, we could then distil to JSON, possibly even without employing a model.
    # - Going only for "whole sequence input" in terms of token length was also a mistake.
    # - A better, more robust and more generalizable approach would be to chunk long protocols into smaller segments,
    # extract actions from each segment, and then merge the results.
