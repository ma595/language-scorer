# Dataset is an iterable but it doesn't implement __iter__
# it falls back to __getitem__


class MyDataset:
    def __getitem__(self, idx):
        if idx >= 3:
            raise IndexError
        return idx * 10

    def __len__(self):
        return 3

ds = MyDataset()

for x in ds:
    print(x)

