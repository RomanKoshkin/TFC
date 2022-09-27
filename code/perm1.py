import torch

@torch.jit.script
def perm(_x):
    x = _x.clone()
    for i in range(x.shape[0]):
        # get a unique vector of sorting indices for each epoch:
        sorting_indices = torch.randint(low=1, high=4, size=(x.shape[2],), dtype=torch.long).sort()[0]
        for ch in range(x.shape[1]):
            # sort each channel in an epoch in the SAME way
            x[i, ch, :] = x[i, ch, sorting_indices]
    return x