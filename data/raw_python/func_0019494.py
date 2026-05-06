def _is_significant(stats, metrics=None):
    """Filter significant motifs based on several statistics.
    
    Parameters
    ----------
    stats : dict
        Statistics disctionary object.

    metrics : sequence
        Metric with associated minimum values. The default is
        (("max_enrichment", 3), ("roc_auc", 0.55), ("enr_at_fpr", 0.55))
    
    Returns
    -------
    significant : bool
    """
    if metrics is None:
        metrics = (("max_enrichment", 3), ("roc_auc", 0.55), ("enr_at_fpr", 0.55))
    
    for stat_name, min_value in metrics:
        if stats.get(stat_name, 0) < min_value:
            return False

    return True