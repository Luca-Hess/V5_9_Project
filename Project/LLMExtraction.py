import openai  # or anthropic, or use local LLM via transformers
import json
import os
import time
from typing import List, Dict

from ModelAnnotation import LabAutomationMachine

# Token for Github API:
token =  os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"

client = openai.OpenAI(
    base_url=endpoint,
    api_key=token
)

class LLMActionExtractor:
    """Extract structured actions from protocols using LLM"""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

        # Define action schema
        self.action_schema = {
            "action": "string (normalized verb, e.g., 'incubate' not 'incubating')",
            "parameters": {
                "container": "string or null",
                "reagent": "string or null",
                "volume": "string or null",
                "temperature": "string or null",
                "duration": "string or null",
                "speed": "string or null",
                "other": "dict of other parameters"
            },
            "automatable": "boolean (true if machine can do this)",
            "confidence": "float (0-1, how clear the instruction is)"
        }

    def create_extraction_prompt(self, protocol_text: str) -> str:
        """Create prompt for LLM to extract structured actions"""
        prompt = f"""You are an expert in laboratory protocols and automation. Extract all actions from the following protocol and structure them for lab automation.

For each action:
1. Normalize the verb (e.g., "incubate", "incubating", "incubation" → "incubate")
2. Extract all parameters (container, reagent, volume, temperature, duration, speed, etc.)
3. Determine if the action can be automated by a liquid handler/incubator/centrifuge
4. Rate your confidence in understanding the instruction

Return a JSON array of actions following this schema:
{json.dumps(self.action_schema, indent=2)}

Automatable actions include: pipette, dispense, aspirate, mix, shake, incubate, heat, cool, centrifuge, wait
Manual actions include: inoculate, count, freeze, thaw, observe, check, prepare (when vague)

Protocol:
{protocol_text}

Respond ONLY with valid JSON, no other text."""

        return prompt

    def extract_actions(self, protocol_text: str) -> List[Dict]:
        """Call LLM to extract actions"""
        prompt = self.create_extraction_prompt(protocol_text)

        try:
            # For OpenAI
            response = client.chat.completions.create(
                model = model,
                messages=[
                    {"role": "system", "content": "You are a laboratory automation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for consistency
            )

            result = response.choices[0].message.content
            actions = json.loads(result)
            return actions

        except Exception as e:
            print(f"LLM extraction failed: {e}")

            if "Too many requests" in str(e):
                print("Rate limit exceeded. Stopping extraction.")
                exit(1)

            return []

    def extract_from_file(self, txt_path: str) -> List[Dict]:
        """Extract actions from a protocol file"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            protocol_text = f.read()

        return self.extract_actions(protocol_text)


class ImprovedAnnotationToScriptConverter:
    """Convert LLM-extracted actions to automation script"""

    def __init__(self):
        self.machine = LabAutomationMachine()

    def generate_script(self, actions: List[Dict]) -> str:
        """Generate automation script from LLM-extracted actions"""
        script_lines = ["# Lab Automation Script", "# Generated from protocol analysis", ""]
        current_mode = None

        for i, action in enumerate(actions):
            automatable = action.get('automatable', False)
            action_name = action.get('action', 'unknown')
            params = action.get('parameters', {})

            # Switch between automated/manual sections
            if automatable != current_mode:
                if automatable:
                    script_lines.append("\n### AUTOMATED SECTION ###")
                else:
                    script_lines.append("\n### MANUAL SECTION ###")
                current_mode = automatable

            # Format action
            if automatable:
                # Clean parameters (remove nulls)
                clean_params = {k: v for k, v in params.items() if v is not None and v != {} and k != 'other'}
                if params.get('other'):
                    clean_params.update(params['other'])

                param_str = ", ".join([f"{k}={repr(v)}" for k, v in clean_params.items()])
                script_lines.append(f"{action_name}({param_str})")
            else:
                # Manual instruction
                param_desc = ", ".join([f"{k}: {v}" for k, v in params.items() if v is not None])
                script_lines.append(f"# MANUAL: {action_name} ({param_desc})")

        return "\n".join(script_lines)


class SyntheticDatasetGenerator:
    """Generate training dataset using LLM extraction"""

    def __init__(self, llm_extractor: LLMActionExtractor, max_per_minute: int = 14):
        self.extractor = llm_extractor
        self.converter = ImprovedAnnotationToScriptConverter()
        self.max_per_minute = max_per_minute
        self.min_interval = 60 / max_per_minute
        self._last_request = 0.0

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()

    def generate_from_protocols(self, data_dir: str):
        """Process all protocols and create synthetic dataset"""

        dataset_complete = []

        for fname in os.listdir(data_dir):
            dataset = []
            if not fname.endswith('.txt'):
                continue

            txt_path = os.path.join(data_dir, fname)
            basename = fname[:-4]
            output_path = f'{data_dir}/{basename}.json'

            if os.path.exists(output_path):
                print(f"Skipping {fname}, {output_path} already exists.")
                continue

            print(f"Processing {fname}...")

            with open(txt_path, 'r', encoding='utf-8') as f:
                protocol_text = f.read().strip()

            # Rate limiting LLM calls
            self._wait_for_rate_limit()

            # Extract actions using LLM
            actions = self.extractor.extract_actions(protocol_text)

            # Skip if no actions extracted
            if actions is None or len(actions) == 0:
                print(f"  No actions extracted for {fname}, skipping.")
                continue

            # Generate automation script
            script = self.converter.generate_script(actions)

            dataset.append({
                'protocol': protocol_text,
                'actions': actions,
                'script': script,
                'source_file': fname
            })

            dataset_complete.append(dataset)

            # Save individual datasets to JSON
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2)
            except Exception as e:
                print(f"  Failed to save {output_path}: {e}")

            # Save combined dataset
            try:
                with open(f"{data_dir}/dataset_complete.json", mode='w', encoding='utf-8') as f:
                    json.dump(dataset_complete, f, indent=2)
            except Exception as e:
                print(f"  Failed to save combined dataset: {e}")

        return dataset_complete

if __name__ == "__main__":
    # Example usage
    extractor = LLMActionExtractor(client, model)
    generator = SyntheticDatasetGenerator(extractor)
    data_path = "WLP-Dataset-master/train"
    generator.generate_from_protocols(data_path)

# Usage example
# def test_llm_extraction():
#     """Test LLM-based extraction on a sample protocol"""
#
#     # Initialize extractor (you'll need an API key)
#     extractor = LLMActionExtractor(client, model)
#
#     sample_protocol = """Radioactive Labeling with T4 PNK (M0201S)
# Set-up the following reaction:.
# Incubate at 37°C for 30 minutes.
# Heat inactivate by incubating at 65°C for 20 minutes."""
#
#     # Extract actions
#     actions = extractor.extract_actions(sample_protocol)
#
#     print("Extracted Actions:")
#     print(json.dumps(actions, indent=2))
#
#     # Generate script
#     converter = ImprovedAnnotationToScriptConverter()
#     script = converter.generate_script(actions)
#
#     print("\n" + "=" * 60)
#     print("Generated Automation Script:")
#     print("=" * 60)
#     print(script)
#
# if __name__ == "__main__":
#     test_llm_extraction()


# # For local/open-source LLM alternative
# class LocalLLMExtractor(LLMActionExtractor):
#     """Use local LLM (e.g., LLaMA, Mistral) instead of API"""
#
#     def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct"):
#         from transformers import pipeline
#
#         self.pipe = pipeline(
#             "text-generation",
#             model=model_name,
#             device_map="auto",
#             max_new_tokens=2048
#         )
#
#     def extract_actions(self, protocol_text: str) -> List[Dict]:
#         prompt = self.create_extraction_prompt(protocol_text)
#
#         result = self.pipe(prompt, do_sample=False)[0]['generated_text']
#
#         # Extract JSON from response
#         try:
#             # Find JSON array in response
#             start = result.find('[')
#             end = result.rfind(']') + 1
#             json_str = result[start:end]
#             actions = json.loads(json_str)
#             return actions
#         except:
#             return []
