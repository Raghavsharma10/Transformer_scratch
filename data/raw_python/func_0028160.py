def group_dict_by_value(d: dict) -> dict:
    """
    Group a dictionary by values.


    Parameters
    ----------
    d : dict
        Input dictionary

    Returns
    -------
    dict
        Output dictionary. The keys are the values of the initial dictionary
        and the values ae given by a list of keys corresponding to the value.

    >>> group_dict_by_value({2: 3, 1: 2, 3: 1})
    {3: [2], 2: [1], 1: [3]}
    >>> group_dict_by_value({2: 3, 1: 2, 3: 1, 10:1, 12: 3})
    {3: [2, 12], 2: [1], 1: [3, 10]}
    """
    d_out = {}
    for k, v in d.items():
        if v in d_out:
            d_out[v].append(k)
        else:
            d_out[v] = [k]
    return d_out