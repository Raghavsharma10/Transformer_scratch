def poly2o_model(params, shape):
    """lmfit 2nd order polynomial model"""
    mx = params["mx"].value
    my = params["my"].value
    mxy = params["mxy"].value
    ax = params["ax"].value
    ay = params["ay"].value
    off = params["off"].value
    bg = np.zeros(shape, dtype=float) + off
    x = np.arange(bg.shape[0]) - bg.shape[0] // 2
    y = np.arange(bg.shape[1]) - bg.shape[1] // 2
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    bg += ax * x**2 + ay * y**2 + mx * x + my * y + mxy * x * y
    return bg