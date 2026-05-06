def _dihed_angle_low(av, bv, cv, deriv):
    """Similar to dihed_cos, but with relative vectors"""
    a = Vector3(9, deriv, av, (0, 1, 2))
    b = Vector3(9, deriv, bv, (3, 4, 5))
    c = Vector3(9, deriv, cv, (6, 7, 8))
    b /= b.norm()
    tmp = b.copy()
    tmp *= dot(a, b)
    a -= tmp
    tmp = b.copy()
    tmp *= dot(c, b)
    c -= tmp
    a /= a.norm()
    c /= c.norm()
    result = dot(a, c).results()
    # avoid trobles with the gradients by either using arccos or arcsin
    if abs(result[0]) < 0.5:
        # if the cosine is far away for -1 or +1, it is safe to take the arccos
        # and fix the sign of the angle.
        sign = 1-(np.linalg.det([av, bv, cv]) > 0)*2
        return _cos_to_angle(result, deriv, sign)
    else:
        # if the cosine is close to -1 or +1, it is better to compute the sine,
        # take the arcsin and fix the sign of the angle
        d = cross(b, a)
        side = (result[0] > 0)*2-1 # +1 means angle in range [-pi/2,pi/2]
        result = dot(d, c).results()
        return _sin_to_angle(result, deriv, side)