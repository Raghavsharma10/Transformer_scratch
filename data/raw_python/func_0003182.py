def _mapping_to_tuple_pairs(d):
    """
        Convert a mapping object (such as a dictionary) to tuple pairs,
        using its keys and values to generate the pairs and then generating
        all possible combinations between those
        e.g. {1: (1,2,3)} -> (((1, 1),), ((1, 2),), ((1, 3),))
    """
    # order the keys, this will prevent different implementations of Python,
    # return different results from the same dictionary since the order of
    # iteration depends on it
    t = []
    ord_keys = sorted(d.keys())
    for k in ord_keys:
        t.append(_product(k, d[k]))
    return tuple(product(*t))