def tilt_residual(params, data, mask):
    """lmfit tilt residuals"""
    bg = tilt_model(params, shape=data.shape)
    res = (data - bg)[mask]
    return res.flatten()