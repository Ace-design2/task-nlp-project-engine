from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

dataset = load_dataset("csv", data_files="data/intents.csv")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=False, truncation=True)

dataset = dataset.map(tokenize, batched=True)

labels = list(set(dataset["train"]["label"]))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir="model_intent",
    per_device_train_batch_size=8,
    num_train_epochs=4,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()
model.save_pretrained("model_intent")
tokenizer.save_pretrained("model_intent")
