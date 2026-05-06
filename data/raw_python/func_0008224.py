def overlap(xl1, yl1, nx1, ny1, xl2, yl2, nx2, ny2):
    """
    Determines whether two windows overlap
    """
    return (xl2 < xl1+nx1 and xl2+nx2 > xl1 and
            yl2 < yl1+ny1 and yl2+ny2 > yl1)