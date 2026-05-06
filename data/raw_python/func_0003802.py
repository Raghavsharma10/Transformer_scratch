def _bend_cos_low(a, b, deriv):
    """Similar to bend_cos, but with relative vectors"""
    a = Vector3(6, deriv, a, (0, 1, 2))
    b = Vector3(6, deriv, b, (3, 4, 5))
    a /= a.norm()
    b /= b.norm()
    return dot(a, b).results()