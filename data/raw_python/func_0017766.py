def num_batches(n, batch_size):
    """Compute the number of mini-batches required to cover a data set of
    size `n` using batches of size `batch_size`.

    Parameters
    ----------
    n: int
        the number of samples in the data set
    batch_size: int
        the mini-batch size

    Returns
    -------
    int: the number of batches required
    """
    b = n // batch_size
    if n % batch_size > 0:
        b += 1
    return b