def softmax(x):
    """Can be replaced once scipy 1.3 is released, although numeric stability should be checked."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:, None]