import numpy as np
import pandas as pd
import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from matplotlib import pyplot as plt


device = "cuda" if cuda.is_available() else "cpu"


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
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train(epoch, tokenizer, model, device, loader, optimizer, model_params):
    model.train()
    epoch_loss = 0
    count = 0
    for _, data in enumerate(
        pbar := tqdm(
            loader, desc="epoch:" + str(epoch), bar_format="{l_bar}{bar:10}{r_bar}"
        ),
        0,
    ):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        pbar.set_description("loss: %s" % loss.item())
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

    epoch_loss /= count
    return epoch_loss


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    total_inputs = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader), 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=512,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
            inputs = [
                tokenizer.decode(
                    i, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for i in ids
            ]
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for t in y
            ]
            total_inputs.extend(inputs)
            predictions.extend(preds)
            actuals.extend(target)
    return total_inputs, predictions, actuals


def T5Trainer(dataframe, source_text, target_text, model_params):
    inference_mode = False

    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    if inference_mode:
        model.from_pretrained("./outputs/model_files")
    model = model.to(device)
    print("[Data]: Reading data...")
    dataframe = dataframe[[source_text, target_text]]
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

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

    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }
    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    if not inference_mode:
        total_losses = []
        for epoch in range(model_params["TRAIN_EPOCHS"]):
            loss = train(
                epoch,
                tokenizer,
                model,
                device,
                training_loader,
                optimizer,
                model_params,
            )
            total_losses.append(loss)

            path = os.path.join(model_params["OUTPUT_MODEL"], "epoch{}".format(epoch))
            os.mkdir(path)

            model.save_pretrained(path)
            tokenizer.save_pretrained(path)

        inputs, predictions, actuals = validate(
            epoch, tokenizer, model, device, val_loader
        )
        final_df = pd.DataFrame(
            {"Input": inputs, "Generated": predictions, "Target": actuals}
        )
        final_df.to_csv(os.path.join(model_params["OUTPUT"], "predictions.csv"))

        plt.plot(total_losses)
        plt.savefig(os.path.join(model_params["OUTPUT"], "loss.png"), dpi=300)


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
os.mkdir(model_params["OUTPUT"])
os.mkdir(model_params["OUTPUT_MODEL"])

path = "./data/clang8.tsv"
# path = "./data/clang8_sample.csv"
df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
df.columns = ["source", "target"]

T5Trainer(
    dataframe=df,
    source_text="source",
    target_text="target",
    model_params=model_params,
)
