def faster_roc(actuals, controls):
    """
    Histogram based implementation of AUC unde ROC curve.
    Parameters:
        actuals : list
            A list of numeric values for positive observations.
        controls : list
            A list of numeric values for negative observations.
    """
    assert(type(actuals) is np.ndarray)
    assert(type(controls) is np.ndarray)

    if len(actuals)<500:
        raise RuntimeError('This method might be incorrect when '+
                'not enough actuals are present. Needs to be checked before '+
                'proceeding. Stopping here for you to do so.')

    actuals = np.ravel(actuals)
    controls = np.ravel(controls)
    if np.isnan(actuals).any():
        raise RuntimeError('NaN found in actuals')
    if np.isnan(controls).any():
        raise RuntimeError('NaN found in controls')

    thresholds = np.hstack([-np.inf, np.unique(actuals), np.inf])+np.finfo(float).eps
    true_pos_rate = np.nan*np.empty(thresholds.size-1)
    false_pos_rate = np.nan*np.empty(thresholds.size-1)
    num_act = float(len(actuals))
    num_ctr = float(len(controls))

    actuals = 1-(np.cumsum(np.histogram(actuals, thresholds)[0])/num_act)
    controls = 1-(np.cumsum(np.histogram(controls, thresholds)[0])/num_ctr)
    true_pos_rate = actuals
    false_pos_rate = controls
    #true_pos_rate = np.concatenate(([0], true_pos_rate, [1]))
    false_pos_rate = false_pos_rate
    auc = -1*np.dot(np.diff(false_pos_rate), true_pos_rate[0:-1])
    # treat cases where TPR of one is not reached before FPR of one
    # by using trapezoidal integration for the last segment
    # (add the missing triangle)
    if false_pos_rate[-2] == 1:
        auc += ((1-true_pos_rate[-3])*.5*(1-false_pos_rate[-3]))
    return (auc, true_pos_rate, false_pos_rate)