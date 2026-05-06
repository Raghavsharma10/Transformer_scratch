def generate_chunks(data, chunk_size=DEFAULT_CHUNK_SIZE):
    """Yield 'chunk_size' items from 'data' at a time."""
    iterator = iter(repeated.getvalues(data))

    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            return

        yield chunk