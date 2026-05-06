def chunker(iterable, size=5, fill=''):
    """Chunk the iterable.

    Parameters
    ----------
    iterable
        A list.

    size
        The size of the chunks.

    fill
        Fill value if the chunk is not of length 'size'.

    Yields
    -------
    chunk
        A chunk of length 'size'.


    Examples
    -------
    >>> l = list(range(6))
    >>> chunks = list(chunker(l, size=4, fill=''))
    >>> chunks == [[0, 1, 2, 3], [4, 5, '', '']]
    True
    """

    for index in range(0, len(iterable) // size + 1):
        to_yield = iterable[index * size: (index + 1) * size]

        # Stop yielding if empty
        if len(to_yield) == 0:
            break

        # Add fill values if there are too few elements
        if len(to_yield) < size:
            yield to_yield + [fill] * (size - len(to_yield))
        else:
            # Yield
            yield to_yield