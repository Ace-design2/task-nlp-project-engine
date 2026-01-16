import os
import sys
import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)


def validate_jsonl(path, max_display=5):
    if not os.path.exists(path):
        print(f"File not found: {os.path.abspath(path)}", file=sys.stderr)
        return False
    bad_examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                bad_examples.append((i + 1, "invalid_json", str(e), line[:200]))
                break

            # Basic expected structure: {"text": str, "labels": [[start,end,entity], ...]}
            labels = obj.get("labels")
            if labels is None:
                bad_examples.append((i + 1, "missing_labels", type(labels).__name__, obj))
                break
            if not isinstance(labels, list):
                bad_examples.append((i + 1, "labels_not_list", type(labels).__name__, labels))
                break
            for j, item in enumerate(labels):
                # expect a 3-element list/tuple [start, end, entity]
                if not (isinstance(item, (list, tuple)) and len(item) == 3):
                    bad_examples.append((i + 1, "label_item_shape", type(item).__name__, item))
                    break
                start, end, entity = item
                if not isinstance(start, int) or not isinstance(end, int):
                    bad_examples.append((i + 1, "label_offsets_not_int", (type(start).__name__, type(end).__name__), item))
                    break
                if not isinstance(entity, str):
                    bad_examples.append((i + 1, "entity_not_str", type(entity).__name__, item))
                    break
            if bad_examples:
                break
    if bad_examples:
        print(f"Found malformed entries in {os.path.abspath(path)}:")
        for ex in bad_examples[:max_display]:
            print(f"  Line {ex[0]}: {ex[1]} -> {ex[2]} | sample: {ex[3]}")
        print("\nFix the file so each label is a [start(int), end(int), entity(str)] triple, one JSON object per line.")
        return False
    return True


TRAIN_PATH = os.path.join("data", "ner_train.jsonl")
VALID_PATH = os.path.join("data", "ner_valid.jsonl")

ok_train = validate_jsonl(TRAIN_PATH)
ok_valid = validate_jsonl(VALID_PATH)
if not (ok_train and ok_valid):
    sys.exit(1)


def load_jsonl_to_dataset(path):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # normalize labels to list of dicts: {start:int, end:int, entity:str}
            normalized = []
            for lbl in obj.get("labels", []):
                if isinstance(lbl, (list, tuple)) and len(lbl) == 3:
                    start, end, entity = lbl
                elif isinstance(lbl, dict):
                    start = lbl.get("start")
                    end = lbl.get("end")
                    entity = lbl.get("entity") or lbl.get("label")
                else:
                    # skip malformed label
                    continue
                try:
                    start = int(start)
                    end = int(end)
                except Exception:
                    # keep as-is; validation should have caught this
                    pass
                normalized.append({"start": start, "end": end, "entity": str(entity)})
            obj["labels"] = normalized
            items.append(obj)
    return Dataset.from_list(items)

dataset = DatasetDict({
    "train": load_jsonl_to_dataset(TRAIN_PATH),
    "validation": load_jsonl_to_dataset(VALID_PATH),
})

labels = ["O", "B-TASK", "I-TASK", "B-DEADLINE", "I-DEADLINE", "B-PRIORITY", "I-PRIORITY", "B-CATEGORY", "I-CATEGORY"]
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align(example):
    # Do not pad here — leave padding to the data collator during batching
    # so that input_ids and labels remain aligned per-example.
    tokens = tokenizer(example["text"], truncation=True, padding=False, return_offsets_mapping=True)
    offsets = tokens["offset_mapping"]
    # initialize with 'O' for all tokens
    labels_out = ["O"] * len(offsets)

    for lbl in example.get("labels", []):
        if isinstance(lbl, dict):
            start = lbl.get("start")
            end = lbl.get("end")
            entity = lbl.get("entity")
        else:
            try:
                start, end, entity = lbl
            except Exception:
                continue

        # ensure types
        if not (isinstance(start, int) and isinstance(end, int) and isinstance(entity, str)):
            continue

        first_token = True
        for i, (o_start, o_end) in enumerate(offsets):
            # skip special tokens (offsets (0,0))
            if o_start == 0 and o_end == 0:
                continue
            # no overlap
            if o_end <= start or o_start >= end:
                continue
            tag = ("B-" if first_token else "I-") + entity
            labels_out[i] = tag
            first_token = False

    # map tag strings to ids, default to 'O' if unknown
    tokens["labels"] = [label2id.get(l, label2id["O"]) for l in labels_out]
    tokens.pop("offset_mapping")
    return tokens

dataset = dataset.map(tokenize_and_align)

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

import inspect

# Define separate directories for training checkpoints and the final model
TRAINING_OUTPUT_DIR = "model_ner_checkpoints"
FINAL_MODEL_DIR = "model_ner"

# Build TrainingArguments kwargs dynamically to support multiple transformers versions
ta_kwargs = {
    "output_dir": TRAINING_OUTPUT_DIR,
    "num_train_epochs": 4,
    "per_device_train_batch_size": 4,
}
ta_init_params = inspect.signature(TrainingArguments.__init__).parameters
if "evaluation_strategy" in ta_init_params:
    ta_kwargs["evaluation_strategy"] = "epoch"

args = TrainingArguments(**ta_kwargs)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
