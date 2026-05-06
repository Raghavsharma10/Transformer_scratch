def merge_all(dcts):
    """
        Shallow merge all the dcts
    :param dcts:
    :return:
    """
    return reduce(
        lambda accum, dct: merge(accum, dct),
        dict(),
        dcts
    )