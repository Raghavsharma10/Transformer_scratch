def poly2o_residual(params, data, mask):
    """lmfit 2nd order polynomial residuals"""
    bg = poly2o_model(params, shape=data.shape)
    res = (data - bg)[mask]
    return res.flatten()