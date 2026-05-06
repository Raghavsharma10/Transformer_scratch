def _bond_length_low(r, deriv):
    """Similar to bond_length, but with a relative vector"""
    r = Vector3(3, deriv, r, (0, 1, 2))
    d = r.norm()
    return d.results()