def sql_pathdist_func(path1, path2, sep=os.path.sep):
    """
    Return a distance between `path1` and `path2`.

    >>> sql_pathdist_func('a/b/', 'a/b/', sep='/')
    0
    >>> sql_pathdist_func('a/', 'a/b/', sep='/')
    1
    >>> sql_pathdist_func('a', 'a/', sep='/')
    0

    """
    seq1 = path1.rstrip(sep).split(sep)
    seq2 = path2.rstrip(sep).split(sep)
    return sum(1 for (p1, p2) in zip_longest(seq1, seq2) if p1 != p2)