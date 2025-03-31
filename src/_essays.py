import torch
from typing import Dict, Any, Tuple
from torch import Tensor
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
import pandas as pd


# create a dataset class that inherits from torch.Dataset
class EssayDataset(Dataset):
    """
    Dataset class for handing training and test data.
    TODO: Generalise to handle validation split as well.


    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokeniser: BertTokenizer,
        target_label=None,
        max_length: int = 512,
    ):
        """
        Args:
        df (pd.DataFrame): Dataframe of train or test dataset.
        tokeniser (BertTokenizer): Pre-trained tokenizer
        target_label (string) : The label corresponding to our scores.
        max_length(int, optional):
        """
        self.data = df
        self.tokenizer = tokeniser
        self.max_length = max_length
        self.target_label = target_label

    def __len__(self) -> int:
        """
        Returns the number of samples.
        To iterate over it it needs to know its length.
        Torch automatically shuffles the dataset after we've visited all samples.
        Therefore it needs to know its length.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """
        Tokenises the essay and returns input or input+label depending on
        whether self.target_label is None.

        Returns:
            data_dict (Dict[str, Tensor]): Dictionary of tokenised essay, attention_mask and score

        """

        # performance implications of this?

        essay = str(self.data.iloc[index]["content"])  # get essay text
        # score = int(self.data.iloc[index]["score"]) - 1 # convert to 0-based index

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

        # this generalises the EssayDataset class to work with test data as well
        # we also need to check that the target_label is indeed "score"
        # obviously this could be generalised further.
        if self.target_label is not None:
            # convert to 0-based index
            data_dict["score"] = torch.tensor(int(self.data.iloc[index]["score"]) - 1)

        return data_dict


def split_essay_data(full_dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """
    Split essay data for to check performance on held out validation set.
    This sits outside the EssayDataset class as it is a utility function.
    Separation of concerns. It decoupled from the Dataset class, it allows
    us to test other strategies for splitting the data.

    This also relies on knowing the length of the dataset.

    Args:
        full_dataset (Dataset): The full dataset that we intend to split.

    Returns:
        train_dataset (Dataset):
        val_dataset (Dataset):

    """
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # set manual seed to ensure reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset
