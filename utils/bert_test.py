import torch
from torch import nn, tensor, randint
from numpy import random
from transformers import BertForSequenceClassification


def test_loss():
    # 
    bs = 10
    cls = 5

    # fix seed for reproducibility 

    # these represent the outputs of our model
    logits = random.rand(10, 5)
    # convert to pytorch tensor
    logits = tensor(logits)

    m = nn.Softmax(dim=1)
    probs = m(tensor(logits))

    # experiment with getting the maximum value using argmax
    probs.argmax(dim=1)

    # create a bs number of 'correct' class labels, numbered between 0 and 4 inclusive
    labels = randint(0, cls, (bs,))

    # correct probs
    correct_probs = probs[range(bs), labels]  # e.g., probs[i, label[i]]

    losses = -torch.log(correct_probs)

    # and then average?


    

def load_model():
    model = BertForSequenceClassification.from_pretrained("../bert-tiny", num_labels=5)
    print(model)


# load_model()


test_loss()
# load the bert model and print
