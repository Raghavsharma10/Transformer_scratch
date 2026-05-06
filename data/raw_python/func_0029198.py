def dict_selective_merge(a, b, selection, path=None):
    """Conditionally merges b into a if b's keys are contained in selection

    :param a:
    :param b:
    :param selection: limit merge to these top-level keys
    :param path:
    :return:
    """
    if path is None:
        path = []
    for key in b:
        if key in selection:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    dict_selective_merge(a[key], b[key], b[key].keys(), path + [str(key)])
                elif a[key] != b[key]:
                    # update the value
                    a[key] = b[key]
            else:
                a[key] = b[key]
    return a