def reduce(reducer, data, chunk_size=DEFAULT_CHUNK_SIZE):
    """Repeatedly call fold and merge on data and then finalize.

    Arguments:
        data: Input for the fold function.
        reducer: The IReducer to use.
        chunk_size: How many items should be passed to fold at a time?

    Returns:
        Return value of finalize.
    """
    if not chunk_size:
        return finalize(reducer, fold(reducer, data))

    # Splitting the work up into chunks allows us to, e.g. reduce a large file
    # without loading everything into memory, while still being significantly
    # faster than repeatedly calling the fold function for every element.
    chunks = generate_chunks(data, chunk_size)
    intermediate = fold(reducer, next(chunks))
    for chunk in chunks:
        intermediate = merge(reducer, intermediate, fold(reducer, chunk))

    return finalize(reducer, intermediate)