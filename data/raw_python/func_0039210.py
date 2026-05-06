def read_next_block(infile, block_size=io.DEFAULT_BUFFER_SIZE):
    """Iterates over the file in blocks."""
    chunk = infile.read(block_size)

    while chunk:
        yield chunk

        chunk = infile.read(block_size)