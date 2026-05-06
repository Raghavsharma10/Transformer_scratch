def get_metrics(tag):
    """
    Return the values for the metrics with the given tag or all the available metrics if None
    """
    if tag is None:
        return metrics.metrics_by_name_list(metrics.metrics())
    else:
        return metrics.metrics_by_tag(tag)