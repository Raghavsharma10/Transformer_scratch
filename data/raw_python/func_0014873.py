def _chunked(iterable, n):
    """
    Collect data into chunks of up to length n.
    :type iterable: Iterable[T]
    :type n: int
    :rtype: Iterator[list[T]]
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if chunk:
            yield chunk
        else:
            return