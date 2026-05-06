def compare (v1, v2):
    """old style __cmp__ function returning -1, 0, 1"""
    v1_norm = normalize(v1)
    v2_norm = normalize(v2)
    if v1_norm < v2_norm:
        return -1
    if v1_norm > v2_norm:
        return 1
    return 0