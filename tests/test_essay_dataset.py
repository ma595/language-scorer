import pytest
import pandas as pd
from transformers import BertTokenizer
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src._essays import EssayDataset  # or whatever you're importing
import torch



@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "content": ["This is a test essay.", "Another sample essay."],
        "score": [2, 4]
    })


@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained("bert-tiny")


def test_dataset_length(sample_df, tokenizer):
    dataset = EssayDataset(sample_df, tokenizer, target_label="score")
    assert len(dataset) == 2


def test_dataset_item_structure(sample_df, tokenizer):
    dataset = EssayDataset(sample_df, tokenizer, target_label="score")
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "score" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert item["input_ids"].shape[0] == 512  # max length
    assert isinstance(item["score"], torch.Tensor)
    assert item["score"].item() == 1  # 2 - 1 due to 0-based index


def test_dataset_without_labels(sample_df, tokenizer):
    dataset = EssayDataset(sample_df.drop(columns=["score"]), tokenizer)
    item = dataset[0]
    assert "score" not in item
    assert "input_ids" in item
    assert item["input_ids"].shape[0] == 512


def test_tokenizer_truncation(sample_df, tokenizer):
    long_text = "word " * 1000  # force truncation
    sample_df.iloc[0]["content"] = long_text
    dataset = EssayDataset(sample_df, tokenizer, target_label="score")
    item = dataset[0]
    assert item["input_ids"].shape[0] == 512

