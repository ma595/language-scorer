import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from typing import Dict


# Dataset class
class EssayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizer,
        target_label=None,
        max_length: int = 512,
    ):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_label = target_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        essay = str(self.data.iloc[index]["content"])
        encoding = self.tokenizer(
            essay,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        data_dict = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

        if self.target_label is not None:
            data_dict["score"] = torch.tensor(int(self.data.iloc[index]["score"]) - 1)

        return data_dict


# Data split utility
def split_essay_data(full_dataset: Dataset):
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    torch.manual_seed(42)
    return random_split(full_dataset, [train_size, val_size])


# Lightning Module
class EssayScorer(pl.LightningModule):
    def __init__(self, model_name="bert-tiny", num_labels=4, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = self.loss_fn(outputs.logits, batch["score"])
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == batch["score"]).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = self.loss_fn(outputs.logits, batch["score"])
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == batch["score"]).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# Prepare data and dataloaders
tokenizer = BertTokenizer.from_pretrained("bert-tiny")
df = pd.read_csv("data/cleaned_dataset.csv")
full_dataset = EssayDataset(df, tokenizer, target_label="score")
train_dataset, val_dataset = split_essay_data(full_dataset)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Train
model = EssayScorer()
trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices="auto")
trainer.fit(model, train_loader, val_loader)
