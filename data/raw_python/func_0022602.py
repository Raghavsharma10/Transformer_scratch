def metric_detail(slug, with_data_table=False):
    """Template Tag to display a metric's *current* detail.

    * ``slug`` -- the metric's unique slug
    * ``with_data_table`` -- if True, prints the raw data in a table.

    """
    r = get_r()
    granularities = list(r._granularities())
    metrics = r.get_metric(slug)
    metrics_data = []
    for g in granularities:
        metrics_data.append((g, metrics[g]))

    return {
        'granularities': [g.title() for g in granularities],
        'slug': slug,
        'metrics': metrics_data,
        'with_data_table': with_data_table,
    }