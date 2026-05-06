def cubic_bucket_warp(x, n, l1, l2, l3, x0, w1, w2, w3):
    """Warps the length scale with a piecewise cubic "bucket" shape.
    
    Parameters
    ----------
    x : float or array-like of float
        Locations to evaluate length scale at.
    n : non-negative int
        Derivative order to evaluate. Only first derivatives are supported.
    l1 : positive float
        Length scale to the left of the bucket.
    l2 : positive float
        Length scale in the bucket.
    l3 : positive float
        Length scale to the right of the bucket.
    x0 : float
        Location of the center of the bucket.
    w1 : positive float
        Width of the left side cubic section.
    w2 : positive float
        Width of the bucket.
    w3 : positive float
        Width of the right side cubic section.
    """
    x1 = x0 - w2 / 2.0 - w1 / 2.0
    x2 = x0 + w2 / 2.0 + w3 / 2.0
    x_shift_1 = (x - x1 + w1 / 2.0) / w1
    x_shift_2 = (x - x2 + w3 / 2.0) / w3
    if n == 0:
        return (
            l1 * (x <= (x1 - w1 / 2.0)) + (
                -2.0 * (l2 - l1) * (x_shift_1**3 - 3.0 / 2.0 * x_shift_1**2) + l1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            l2 * ((x >= (x1 + w1 / 2.0)) & (x <= x2 - w3 / 2.0)) + (
                -2.0 * (l3 - l2) * (x_shift_2**3 - 3.0 / 2.0 * x_shift_2**2) + l2
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0))) +
            l3 * (x >= (x2 + w3 / 2.0))
        )
    elif n == 1:
        return (
            (
                -2.0 * (l2 - l1) * (3 * x_shift_1**2 - 3.0 * x_shift_1) / w1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            (
                -2.0 * (l3 - l2) * (3 * x_shift_2**2 - 3.0 * x_shift_2) / w3
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0)))
        )
    else:
        raise NotImplementedError("Only up to first derivatives are supported!")