def _opbend_cos_low(a, b, c, deriv):
    """Similar to opbend_cos, but with relative vectors"""
    a = Vector3(9, deriv, a, (0, 1, 2))
    b = Vector3(9, deriv, b, (3, 4, 5))
    c = Vector3(9, deriv, c, (6, 7, 8))
    n  = cross(a,b)
    n /= n.norm()
    c /= c.norm()
    temp = dot(n,c)
    result = temp.copy()
    result.v = np.sqrt(1.0-temp.v**2)
    if result.deriv > 0:
        result.d *= -temp.v
        result.d /= result.v
    if result.deriv > 1:
        result.dd *= -temp.v
        result.dd /= result.v
        temp2 = np.array([temp.d]).transpose()*temp.d
        temp2 /= result.v**3
        result.dd -= temp2
    return result.results()