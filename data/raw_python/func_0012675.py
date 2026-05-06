def nss_model(prediction, fm):
    """
    wraps nss functionality for model evaluation

    input:
        prediction: 2D matrix
            the model salience map
        fm : fixmat
            Fixations that define the actuals
    """
    (r_y, r_x) = calc_resize_factor(prediction, fm.image_size)
    fix = ((np.array(fm.y-1)*r_y).astype(int),
                            (np.array(fm.x-1)*r_x).astype(int))
    return nss(prediction, fix)