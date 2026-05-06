def normalize_listargument(arg):
    """Check if arg is an iterable (list, tuple, set, dict, np.ndarray, except
        string!). If not, make a list of it. Numpy arrays are flattened and
        converted to lists."""
    if isinstance(arg, np.ndarray):
        return arg.flatten()
    if isinstance(arg, str):
        return [arg]
    if isinstance(arg, list) or isinstance(arg, tuple) or isinstance(arg, dict) or isinstance(arg, set):
        return list(arg)
    return [arg]