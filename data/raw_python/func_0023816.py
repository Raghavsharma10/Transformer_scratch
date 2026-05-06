def h(gbm, array_or_frame, indices_or_columns = 'all'):
    """
    PURPOSE

    Compute Friedman and Popescu's H statistic, in order to look for an interaction in the passed gradient-boosting
    model among the variables represented by the elements of the passed array or frame and specified by the passed
    indices or columns.

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

    The H statistic of the variables or NaN if the computation is spoiled by weak main effects and rounding errors.

    H varies from 0 to 1. The larger H, the stronger the evidence for an interaction among the variables.


    EXAMPLES

    Friedman and Popescu's (2008) formulas (44) and (46) correspond to

        h(F, x, [j, k])

    and

        h(F, x, [j, k, l])

    respectively.


    NOTES

    1. Per Friedman and Popescu, only variables with strong main effects should be examined for interactions. Strengths 
    of main effects are available as gbm.feature_importances_ once gbm has been fitted.

    2. Per Friedman and Popescu, collinearity among variables can lead to interactions in gbm that are not present in
    the target function. To forestall such spurious interactions, check for strong correlations among variables before
    fitting gbm.
    """

    if indices_or_columns == 'all':
        if gbm.max_depth < array_or_frame.shape[1]:
            raise \
                Exception(
                    "gbm.max_depth == {} < array_or_frame.shape[1] == {}, so indices_or_columns must not be 'all'."
                    .format(gbm.max_depth, array_or_frame.shape[1])
                )
    else:
        if gbm.max_depth < len(indices_or_columns):
            raise \
                Exception(
                    "gbm.max_depth == {}, so indices_or_columns must contain at most {} {}."
                    .format(gbm.max_depth, gbm.max_depth, "element" if gbm.max_depth == 1 else "elements")
                )
    check_args_contd(array_or_frame, indices_or_columns)

    arr, model_inds = get_arr_and_model_inds(array_or_frame, indices_or_columns)

    width = arr.shape[1]
    f_vals = {}
    for n in range(width, 0, -1):
        for inds in itertools.combinations(range(width), n):
            f_vals[inds] = compute_f_vals(gbm, model_inds, arr, inds)

    return compute_h_val(f_vals, arr, tuple(range(width)))