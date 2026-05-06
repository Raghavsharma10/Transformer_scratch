def _opbend_angle_low(a, b, c, deriv=0):
    """Similar to opbend_angle, but with relative vectors"""
    result = _opbend_cos_low(a, b, c, deriv)
    sign = np.sign(np.linalg.det([a, b, c]))
    return _cos_to_angle(result, deriv, sign)