# when you can't fit into memory. 
# doesn't have a __getitem__ function can't index it 
# 

from torch.utils.data import IterableDataset, DataLoader
import torch

class MyStreamingDataset(IterableDataset):
    def __init__(self, batch_size=2):
        self.data = list(range(10))
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size]
            yield torch.tensor(batch)

ds = MyStreamingDataset(batch_size=3)
dl = DataLoader(ds)

for batch in dl:
    print(batch)

