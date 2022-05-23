import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class MyDataset(Dataset):
    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "label_ids": target_ids.to(dtype=torch.long),
        }


def compute_metrics(p):
    print(p[0].shape(), p[1].shape())
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall}
    f1 = f1_score(y_true=labels, y_pred=pred)


POST_FIX = "second"
model_params = {
    "MODEL": "t5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "OUTPUT": "./outputs/model_" + POST_FIX,
    "OUTPUT_MODEL": "./outputs/model_" + POST_FIX + "/models",
}

# Read Data
# path = "./data/clang8.tsv"
path = "./data/clang8_sample.csv"

source_text = "source"
target_text = "target"

df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
df.columns = ["source", "target"]
dataframe = df[["source", "target"]]
train_size = 0.8
train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

tokenizer = T5Tokenizer.from_pretrained(
    model_params["MODEL"], model_max_length=model_params["MAX_SOURCE_TEXT_LENGTH"]
)
model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])

training_set = MyDataset(
    train_dataset,
    tokenizer,
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
    source_text,
    target_text,
)
val_set = MyDataset(
    val_dataset,
    tokenizer,
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
    source_text,
    target_text,
)

args = TrainingArguments(
    output_dir="outputs/" + str(datetime.datetime.now()),
    evaluation_strategy="epoch",
    per_device_train_batch_size=model_params["TRAIN_BATCH_SIZE"],
    per_device_eval_batch_size=model_params["VALID_BATCH_SIZE"],
    num_train_epochs=3,
    seed=0,
    eval_accumulation_steps=4,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=training_set,
    eval_dataset=val_set,
)

trainer.train()
