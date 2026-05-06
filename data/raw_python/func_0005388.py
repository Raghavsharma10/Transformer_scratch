def in_batches(iterable, batch_size):
    # type: (Iterable[Any]) -> Generator[List[Any]]
    """ Split the given iterable into batches.

    Args:
        iterable (Iterable[Any]):
            The iterable you want to split into batches.
        batch_size (int):
            The size of each bach. The last batch will be probably smaller (if
            the number of elements cannot be equally divided.

    Returns:
        Generator[list[Any]]: Will yield all items in batches of **batch_size**
            size.

    Example:

        >>> from peltak.core import util
        >>>
        >>> batches = util.in_batches([1, 2, 3, 4, 5, 6, 7], 3)
        >>> batches = list(batches)     # so we can query for lenght
        >>> len(batches)
        3
        >>> batches
        [[1, 2, 3], [4, 5, 6], [7]]

    """
    items = list(iterable)
    size = len(items)

    for i in range(0, size, batch_size):
        yield items[i:min(i + batch_size, size)]