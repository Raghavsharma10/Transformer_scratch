def sizes(count, offset=0, max_chunk=500):
    """
    Helper to iterate over remote data via count & offset pagination.
    """
    if count is None:
        chunk = max_chunk
        while True:
            yield chunk, offset
            offset += chunk
    else:
        while count:
            chunk = min(count, max_chunk)
            count = max(0, count - max_chunk)
            yield chunk, offset
            offset += chunk