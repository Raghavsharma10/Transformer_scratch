def _flatten(l):
    """helper to flatten a list of lists
    """
    res = []
    for sublist in l:
        if isinstance(sublist, whaaaaat.Separator):
            res.append(sublist)
        else:
            for item in sublist:
                res.append(item)
    return res