def roc_model(prediction, fm, ctr_loc = None, ctr_size = None):
    """
    wraps roc functionality for model evaluation

    Parameters:
        prediction: 2D array
            the model salience map
        fm : fixmat
            Fixations that define locations of the actuals
        ctr_loc : tuple of (y.x) coordinates, optional
            Allows to specify control points for spatial
            bias correction
        ctr_size : two element tuple, optional
            Specifies the assumed image size of the control locations,
            defaults to fm.image_size
     """

    # check if prediction is a valid numpy array
    assert type(prediction) == np.ndarray
    # check whether scaling preserved aspect ratio
    (r_y, r_x) = calc_resize_factor(prediction, fm.image_size)
    # read out values in the fdm at actual fixation locations
    # .astype(int) floors numbers in np.array
    y_index = (r_y * np.array(fm.y-1)).astype(int)
    x_index = (r_x * np.array(fm.x-1)).astype(int)
    actuals = prediction[y_index, x_index]
    if not ctr_loc:
        xc = np.random.randint(0, prediction.shape[1], 1000)
        yc = np.random.randint(0, prediction.shape[0], 1000)
        ctr_loc = (yc.astype(int), xc.astype(int))
    else:
        if not ctr_size:
            ctr_size = fm.image_size
        else:
            (r_y, r_x) = calc_resize_factor(prediction, ctr_size)
        ctr_loc = ((r_y * np.array(ctr_loc[0])).astype(int),
                   (r_x * np.array(ctr_loc[1])).astype(int))
    controls = prediction[ctr_loc[0], ctr_loc[1]]
    return fast_roc(actuals, controls)[0]