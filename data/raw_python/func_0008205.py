def chunk_sequence(sequence, chunk_length):
    """Yield successive n-sized chunks from l."""
    for index in range(0, len(sequence), chunk_length):
        yield sequence[index:index + chunk_length]