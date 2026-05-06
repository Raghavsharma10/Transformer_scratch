def _opdist_low(av, bv, cv, deriv):
    """Similar to opdist, but with relative vectors"""
    a = Vector3(9, deriv, av, (0, 1, 2))
    b = Vector3(9, deriv, bv, (3, 4, 5))
    c = Vector3(9, deriv, cv, (6, 7, 8))
    n  = cross(a, b)
    n /= n.norm()
    dist = dot(c, n)
    return dist.results()