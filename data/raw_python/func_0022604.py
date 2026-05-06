def aggregate_detail(slug_list, with_data_table=False):
    """Template Tag to display multiple metrics.

    * ``slug_list`` -- A list of slugs to display
    * ``with_data_table`` -- if True, prints the raw data in a table.

    """
    r = get_r()
    metrics_data = []
    granularities = r._granularities()

    # XXX converting granularties into their key-name for metrics.
    keys = ['seconds', 'minutes', 'hours', 'day', 'week', 'month', 'year']
    key_mapping = {gran: key for gran, key in zip(GRANULARITIES, keys)}
    keys = [key_mapping[gran] for gran in granularities]

    # Our metrics data is of the form:
    #
    #   (slug, {time_period: value, ... }).
    #
    # Let's convert this to (slug, list_of_values) so that the list of
    # values is in the same order as the granularties
    for slug, data in r.get_metrics(slug_list):
        values = [data[t] for t in keys]
        metrics_data.append((slug, values))

    return {
        'chart_id': "metric-aggregate-{0}".format("-".join(slug_list)),
        'slugs': slug_list,
        'metrics': metrics_data,
        'with_data_table': with_data_table,
        'granularities': [g.title() for g in keys],
    }