def profile_poly2o(data, mask):
    """Fit a 2D 2nd order polynomial to `data[mask]`"""
    # lmfit
    params = lmfit.Parameters()
    params.add(name="mx", value=0)
    params.add(name="my", value=0)
    params.add(name="mxy", value=0)
    params.add(name="ax", value=0)
    params.add(name="ay", value=0)
    params.add(name="off", value=np.average(data[mask]))
    fr = lmfit.minimize(poly2o_residual, params, args=(data, mask))
    bg = poly2o_model(fr.params, data.shape)
    return bg