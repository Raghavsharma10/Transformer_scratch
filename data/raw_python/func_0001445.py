def profile_tilt(data, mask):
    """Fit a 2D tilt to `data[mask]`"""
    params = lmfit.Parameters()
    params.add(name="mx", value=0)
    params.add(name="my", value=0)
    params.add(name="off", value=np.average(data[mask]))
    fr = lmfit.minimize(tilt_residual, params, args=(data, mask))
    bg = tilt_model(fr.params, data.shape)
    return bg