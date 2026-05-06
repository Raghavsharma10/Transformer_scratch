def entropy_variance(data, class_attr=None,
    method=DEFAULT_CONTINUOUS_METRIC):
    """
    Calculates the variance fo a continuous class attribute, to be used as an
    entropy metric.
    """
    assert method in CONTINUOUS_METRICS, "Unknown entropy variance metric: %s" % (method,)
    assert (class_attr is None and isinstance(data, dict)) \
        or (class_attr is not None and isinstance(data, list))
    if isinstance(data, dict):
        lst = data
    else:
        lst = [record.get(class_attr) for record in data]
    return get_variance(lst)