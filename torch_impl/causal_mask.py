import torch


def causal_mask(T):
    """This function return the lower traingular matrix
    which is necessary to implement the causal masking - not allwoing the model to see the future tokens"""


    return torch.tril(torch.ones(T,T))
