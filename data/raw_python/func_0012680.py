def emd_model(prediction, fm):
    """
    wraps emd functionality for model evaluation

    requires:
        OpenCV python bindings

    input:
        prediction: the model salience map
        fm : fixmat filtered for the image corresponding to the prediction
    """
    (_, r_x) = calc_resize_factor(prediction, fm.image_size)
    gt = fixmat.compute_fdm(fm, scale_factor = r_x)
    return emd(prediction, gt)