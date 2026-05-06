def interval_overlap(a, b, x, y):
    """Returns by how much two intervals overlap

    assumed that a <= b and x <= y"""
    if b <= x or a >= y:
        return 0
    elif x <= a <= y:
        return min(b, y) - a
    elif x <= b <= y:
        return b - max(a, x)
    elif a >= x and b <= y:
        return b - a
    else:
        assert False