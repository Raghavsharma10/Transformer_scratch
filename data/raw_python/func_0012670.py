def kldiv_model(prediction, fm):
    """
    wraps kldiv functionality for model evaluation

    input:
        prediction: 2D matrix
            the model salience map
        fm : fixmat
            Should be filtered for the image corresponding to the prediction
    """
    (_, r_x) = calc_resize_factor(prediction, fm.image_size)
    q = np.array(prediction, copy=True)
    q -= np.min(q.flatten())
    q /= np.sum(q.flatten())
    return kldiv(None, q, distp = fm, scale_factor = r_x)