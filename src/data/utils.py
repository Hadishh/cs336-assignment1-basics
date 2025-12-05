import torch
import numpy as np


def data_loading(x, batch_size, context_length, device="cpu"):
    starts = np.random.randint(
        low=0, high=x.shape[0] - context_length, size=(batch_size,)
    )

    x_sub = [x[s : s + context_length] for s in starts]
    y = [x[s + 1 : s + 1 + context_length] for s in starts]
    positions = [np.arange(start=s, stop=s + context_length) for s in starts]

    x_sub = np.stack(x_sub)
    y = np.stack(y)
    positions = np.stack(positions)

    x_sub = torch.tensor(x_sub, device=device)
    y = torch.tensor(y, device=device)

    return x_sub, y


if __name__ == "__main__":
    from tests.test_data import test_get_batch

    test_get_batch()
