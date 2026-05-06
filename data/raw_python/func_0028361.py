def at_index(iterable, index):
    # type: (Iterable[T], int) -> T
    """" Return the item at the index of this iterable or raises IndexError.

        WARNING: this will consume generators.

        Negative indices are allowed but be aware they will cause n items to
        be held in memory, where n = abs(index)
    """
    try:
        if index < 0:
            return deque(iterable, maxlen=abs(index)).popleft()

        return next(itertools.islice(iterable, index, index + 1))
    except (StopIteration, IndexError) as e:
        raise_from(IndexError('Index "%d" out of range' % index), e)