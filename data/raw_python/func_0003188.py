def __precision(y_true, y_pred):
    '''
        Precision metric tolerant to unlabeled data in y_true,
        NA values are ignored for the precision calculation
    '''
    # make copies of the arrays to avoid modifying the original ones
    y_true = np.copy(y_true)
    y_pred = np.copy(y_pred)

    # precision = tp/(tp+fp)
    # True nehatives do not affect precision value, so for every missing
    # value in y_true, replace it with 0 and also replace the value
    # in y_pred with 0
    is_nan = np.isnan(y_true)
    y_true[is_nan] = 0
    y_pred[is_nan] = 0
    precision = precision_score(y_true, y_pred)
    return precision