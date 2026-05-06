def _trim_batch(batch, length):
    """Trim the mini-batch `batch` to the size `length`.

    `batch` can be:
    - a NumPy array, in which case it's first axis will be trimmed to size
      `length`
    - a tuple, in which case `_trim_batch` applied recursively to
      each element and the resulting tuple returned

    As a consequence, mini-batches can be structured; lists and tuples can
    be nested arbitrarily deep.

    Parameters
    ----------
    batch: tuple or NumPy array
        the mini-batch to trim
    length: int
        the size to which `batch` is to be trimmed

    Returns
    -------
    tuple or NumPy array of same structure as `batch`
    The trimmed mini-batch
    """
    if isinstance(batch, tuple):
        return tuple([_trim_batch(b, length) for b in batch])
    else:
        return batch[:length]