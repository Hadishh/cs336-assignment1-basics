import torch
import numpy as np
import math
import random


def data_loading(x, batch_size, context_length, device="cpu"):
    starts = np.random.randint(
        low=0, high=x.shape[0] - context_length, size=(batch_size,)
    )

    x_sub = [x[s : s + context_length] for s in starts]
    y = [x[s + 1 : s + 1 + context_length] for s in starts]

    x_sub = np.stack(x_sub)
    y = np.stack(y)

    x_sub = torch.tensor(x_sub, dtype=torch.int, device=device)
    y = torch.tensor(y, dtype=torch.int, device=device)

    return x_sub, y


def valid_data_loading(x, batch_size, context_length, device="cpu", shuffle=True):

    xy = [
        (x[s : s + context_length], x[s + 1 : s + 1 + context_length])
        for s in range(0, x.shape[0] - context_length)
    ]

    if shuffle:
        random.shuffle(xy)

    batches = list()
    for i in range(0, len(xy), batch_size):
        x_sub = [pair[0] for pair in xy[i : i + batch_size]]
        y = [pair[1] for pair in xy[i : i + batch_size]]

        x_sub = np.stack(x_sub)
        y = np.stack(y)

        x_sub = torch.tensor(x_sub, dtype=torch.int, device=device)
        y = torch.tensor(y, dtype=torch.int, device=device)

        batches.append((x_sub, y))

    return batches


if __name__ == "__main__":
    from tests.test_data import test_get_batch

    test_get_batch()
