def correlation_model(prediction, fm):
    """
    wraps numpy.corrcoef functionality for model evaluation

    input:
        prediction: 2D Matrix
            the model salience map
        fm: fixmat
            Used to compute a FDM to which the prediction is compared.
    """
    (_, r_x) = calc_resize_factor(prediction, fm.image_size)
    fdm = compute_fdm(fm, scale_factor = r_x)
    return np.corrcoef(fdm.flatten(), prediction.flatten())[0,1]