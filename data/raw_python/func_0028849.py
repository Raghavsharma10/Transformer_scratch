def join(strin, items):
    """
        Ramda implementation of join
    :param strin:
    :param items:
    :return:
    """
    return strin.join(map(lambda item: str(item), items))