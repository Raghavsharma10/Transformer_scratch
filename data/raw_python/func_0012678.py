def fast_roc(actuals, controls):
    """
    approximates the area under the roc curve for sets of actuals and controls.
    Uses all values appearing in actuals as thresholds and lower sum
    interpolation. Also returns arrays of the true positive rate and the false
    positive rate that can be used for plotting the roc curve.

    Parameters:
        actuals : list
            A list of numeric values for positive observations.
        controls : list
            A list of numeric values for negative observations.
    """
    assert(type(actuals) is np.ndarray)
    assert(type(controls) is np.ndarray)

    actuals = np.ravel(actuals)
    controls = np.ravel(controls)
    if np.isnan(actuals).any():
        raise RuntimeError('NaN found in actuals')
    if np.isnan(controls).any():
        raise RuntimeError('NaN found in controls')

    thresholds = np.hstack([-np.inf, np.unique(actuals), np.inf])[::-1]
    true_pos_rate = np.empty(thresholds.size)
    false_pos_rate = np.empty(thresholds.size)
    num_act = float(len(actuals))
    num_ctr = float(len(controls))

    for i, value in enumerate(thresholds):
        true_pos_rate[i] = (actuals >= value).sum() / num_act
        false_pos_rate[i] = (controls >= value).sum() / num_ctr
    auc = np.dot(np.diff(false_pos_rate), true_pos_rate[0:-1])
    # treat cases where TPR of one is not reached before FPR of one
    # by using trapezoidal integration for the last segment
    # (add the missing triangle)
    if false_pos_rate[-2] == 1:
        auc += ((1-true_pos_rate[-3])*.5*(1-false_pos_rate[-3]))
    return (auc, true_pos_rate, false_pos_rate)