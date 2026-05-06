def expandParameters(*args):
    """Expands parameters (presented as tuples of lists and symbolic names)
    so that each is returned in a new list where each contains the same number
    of values.

    Each `arg` is a tuple containing two items: a list of values and a
    symbolic name.
    """
    count = 1
    for arg in args:
        count = max(len(arg[0]), count)
    results = []
    for arg in args:
        results.append(expandValues(arg[0], count, args[1]))
    return tuple(results)