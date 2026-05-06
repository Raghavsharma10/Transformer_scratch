def bootstrap_auc(df, col, pred_col, n_bootstrap=1000):
    """
    Calculate the boostrapped AUC for a given col trying to predict a pred_col.

    Parameters
    ----------
    df : pandas.DataFrame
    col : str
        column to retrieve the values from
    pred_col : str
        the column we're trying to predict
    n_boostrap : int
        the number of bootstrap samples

    Returns
    -------
    list : AUCs for each sampling
    """
    scores = np.zeros(n_bootstrap)
    old_len = len(df)
    df.dropna(subset=[col], inplace=True)
    new_len = len(df)
    if new_len < old_len:
        logger.info("Dropping NaN values in %s to go from %d to %d rows" % (col, old_len, new_len))
    preds = df[pred_col].astype(int)
    for i in range(n_bootstrap):
        sampled_counts, sampled_pred = resample(df[col], preds)
        if is_single_class(sampled_pred, col=pred_col):
            continue
        scores[i] = roc_auc_score(sampled_pred, sampled_counts)
    return scores