def _bend_angle_low(a, b, deriv):
    """Similar to bend_angle, but with relative vectors"""
    result = _bend_cos_low(a, b, deriv)
    return _cos_to_angle(result, deriv)