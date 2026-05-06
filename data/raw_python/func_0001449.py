def tilt_model(params, shape):
    """lmfit tilt model"""
    mx = params["mx"].value
    my = params["my"].value
    off = params["off"].value
    bg = np.zeros(shape, dtype=float) + off
    x = np.arange(bg.shape[0]) - bg.shape[0] // 2
    y = np.arange(bg.shape[1]) - bg.shape[1] // 2
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    bg += mx * x + my * y
    return bg