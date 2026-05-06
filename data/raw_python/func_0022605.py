def aggregate_history(slugs, granularity="daily", since=None, with_data_table=False):
    """Template Tag to display history for multiple metrics.

    * ``slug_list`` -- A list of slugs to display
    * ``granularity`` -- the granularity: seconds, minutes, hourly,
                         daily, weekly, monthly, yearly
    * ``since`` -- a datetime object or a string string matching one of the
      following patterns: "YYYY-mm-dd" for a date or "YYYY-mm-dd HH:MM:SS" for
      a date & time.
    * ``with_data_table`` -- if True, prints the raw data in a table.

    """
    r = get_r()
    slugs = list(slugs)

    try:
        if since and len(since) == 10:  # yyyy-mm-dd
            since = datetime.strptime(since, "%Y-%m-%d")
        elif since and len(since) == 19:  # yyyy-mm-dd HH:MM:ss
            since = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        # assume we got a datetime object or leave since = None
        pass

    history = r.get_metric_history_chart_data(
        slugs=slugs,
        since=since,
        granularity=granularity
    )

    return {
        'chart_id': "metric-aggregate-history-{0}".format("-".join(slugs)),
        'slugs': slugs,
        'since': since,
        'granularity': granularity,
        'metric_history': history,
        'with_data_table': with_data_table,
    }