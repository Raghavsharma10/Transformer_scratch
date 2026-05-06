def find_all_iter( source, substring, start=None, end=None, overlap=False ):
    """Iterate through every location a substring can be found in a source string.

    source
        The source string to search.

    start
        Start offset to read from (default: start)

    end
        End offset to stop reading at (default: end)

    overlap
        Whether to return overlapping matches (default: false)
    """
    data = source
    base = 0
    if end is not None:
        data = data[:end]
    if start is not None:
        data = data[start:]
        base = start
    pointer = 0
    increment = 1 if overlap else (len( substring ) or 1)
    while True:
        pointer = data.find( substring, pointer )
        if pointer == -1:
            return
        yield base+pointer
        pointer += increment