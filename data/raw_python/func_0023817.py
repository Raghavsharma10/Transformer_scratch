def h_all_pairs(gbm, array_or_frame, indices_or_columns = 'all'):
    """
    PURPOSE

    Compute Friedman and Popescu's two-variable H statistic, in order to look for an interaction in the passed gradient-
    boosting model between each pair of variables represented by the elements of the passed array or frame and specified
    by the passed indices or columns.

    See Jerome H. Friedman and Bogdan E. Popescu, 2008, "Predictive learning via rule ensembles", Ann. Appl. Stat.
    2:916-954, http://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908046, s. 8.1.


    ARGUMENTS

    gbm should be a scikit-learn gradient-boosting model (instance of sklearn.ensemble.GradientBoostingClassifier or
    sklearn.ensemble.GradientBoostingRegressor) that has been fitted to array_or_frame (and a target, not used here).

    array_or_frame should be a two-dimensional NumPy array or a pandas data frame (instance of numpy.ndarray or pandas
    .DataFrame).

    indices_or_columns is optional, with default value 'all'. It should be 'all' or a list of indices of columns of
    array_or_frame if array_or_frame is a NumPy array or a list of columns of array_or_frame if array_or_frame is a
    pandas data frame. If it is 'all', then all columns of array_or_frame are used.


    RETURNS

    A dict whose keys are pairs (2-tuples) of indices or columns and whose values are the H statistic of the pairs of
    variables or NaN if a computation is spoiled by weak main effects and rounding errors.

    H varies from 0 to 1. The larger H, the stronger the evidence for an interaction between a pair of variables.


    EXAMPLE

    Friedman and Popescu's (2008) formula (44) for every j and k corresponds to

        h_all_pairs(F, x)


    NOTES

    1. Per Friedman and Popescu, only variables with strong main effects should be examined for interactions. Strengths 
    of main effects are available as gbm.feature_importances_ once gbm has been fitted.

    2. Per Friedman and Popescu, collinearity among variables can lead to interactions in gbm that are not present in
    the target function. To forestall such spurious interactions, check for strong correlations among variables before
    fitting gbm.
    """

    if gbm.max_depth < 2:
        raise Exception("gbm.max_depth must be at least 2.")
    check_args_contd(array_or_frame, indices_or_columns)

    arr, model_inds = get_arr_and_model_inds(array_or_frame, indices_or_columns)

    width = arr.shape[1]
    f_vals = {}
    for n in [2, 1]:
        for inds in itertools.combinations(range(width), n):
            f_vals[inds] = compute_f_vals(gbm, model_inds, arr, inds)

    h_vals = {}
    for inds in itertools.combinations(range(width), 2):
        h_vals[inds] = compute_h_val(f_vals, arr, inds)
    if indices_or_columns != 'all':
        h_vals = {tuple(model_inds[(inds,)]): h_vals[inds] for inds in h_vals.keys()}
    if not isinstance(array_or_frame, np.ndarray):
        all_cols = array_or_frame.columns.values
        h_vals = {tuple(all_cols[(inds,)]): h_vals[inds] for inds in h_vals.keys()}

    return h_vals