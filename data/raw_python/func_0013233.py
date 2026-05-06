def auprc(y_true, y_pred):
    """Area under the precision-recall curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    return skm.average_precision_score(y_true, y_pred)