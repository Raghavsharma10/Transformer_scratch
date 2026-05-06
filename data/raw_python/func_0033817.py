def group(self, lst, n):
    """
    Group an iterable into an n-tuples iterable. Incomplete
    tuples are discarded
    """
    return itertools.izip(*[itertools.islice(lst, i, None, n) for i in range(n)])