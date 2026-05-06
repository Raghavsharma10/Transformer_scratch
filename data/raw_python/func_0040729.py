def chunks(data, size):
    """
    Generator that splits the given data into chunks
    """
    for i in range(0, len(data), size):
        yield data[i:i + size]