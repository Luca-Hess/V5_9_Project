import os
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch
import tqdm
from torch.optim import AdamW

from peft import LoraConfig, get_peft_model

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
        text, ann = self.examples[idx]
        inputs = self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length",
                                return_tensors="pt")
        labels = self.tokenizer(ann, max_length=self.max_len, truncation=True, padding="max_length",
                                return_tensors="pt")
        inputs["labels"] = labels["input_ids"]
        return {key: val.squeeze(0) for key, val in inputs.items()}

tokenizer = T5Tokenizer.from_pretrained("t5-base")
base_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Configure LoRA
lora_config = LoraConfig(
    r=8,                # rank
    lora_alpha=32,      # scaling
    target_modules=["q", "v"],  # attention projection layers
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(base_model, lora_config)


path = "WLP-Dataset-master/train"

train_dataset = ProtocolDataset(path, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-4)


for epoch in range(3):
    model.train()
    loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch} finished")


def annotate_protocol(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(annotate_protocol("Weigh 5.73 g of TCEP."))
