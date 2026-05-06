def metrics_by_name_list(names):
    """
    Return a dictionary with {metric name: metric value} for all the metrics with the given names.
    """
    results = {}

    for name in names:
        # no lock - a metric could have been removed in the meanwhile
        try:
            results[name] = get(name)
        except InvalidMetricError:
            continue

    return results