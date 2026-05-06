def metric_history(slug, granularity="daily", since=None, to=None,
                   with_data_table=False):
    """Template Tag to display a metric's history.

    * ``slug`` -- the metric's unique slug
    * ``granularity`` -- the granularity: daily, hourly, weekly, monthly, yearly
    * ``since`` -- a datetime object or a string string matching one of the
      following patterns: "YYYY-mm-dd" for a date or "YYYY-mm-dd HH:MM:SS" for
      a date & time.
    * ``to`` -- the date until which we start pulling metrics
    * ``with_data_table`` -- if True, prints the raw data in a table.

    """
    r = get_r()
    try:
        if since and len(since) == 10:  # yyyy-mm-dd
            since = datetime.strptime(since, "%Y-%m-%d")
        elif since and len(since) == 19:  # yyyy-mm-dd HH:MM:ss
            since = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")

        if to and len(to) == 10:  # yyyy-mm-dd
            to = datetime.strptime(since, "%Y-%m-%d")
        elif to and len(to) == 19:  # yyyy-mm-dd HH:MM:ss
            to = datetime.strptime(to, "%Y-%m-%d %H:%M:%S")

    except (TypeError, ValueError):
        # assume we got a datetime object or leave since = None
        pass

    metric_history = r.get_metric_history(
        slugs=slug,
        since=since,
        to=to,
        granularity=granularity
    )

    return {
        'since': since,
        'to': to,
        'slug': slug,
        'granularity': granularity,
        'metric_history': metric_history,
        'with_data_table': with_data_table,
    }