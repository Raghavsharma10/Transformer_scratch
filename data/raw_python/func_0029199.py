def dict_merge(a, b, path=None):
    """merges b into a"""
    return dict_selective_merge(a, b, b.keys(), path)