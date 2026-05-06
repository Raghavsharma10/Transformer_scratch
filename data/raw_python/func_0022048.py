def request_time_facet(field, time_filter, time_gap, time_limit=100):
    """
    time facet query builder
    :param field: map the query to this field.
    :param time_limit: Non-0 triggers time/date range faceting. This value is the maximum number of time ranges to
    return when a.time.gap is unspecified. This is a soft maximum; less will usually be returned.
    A suggested value is 100.
    Note that a.time.gap effectively ignores this value.
    See Solr docs for more details on the query/response format.
    :param time_filter: From what time range to divide by a.time.gap into intervals.
    Defaults to q.time and otherwise 90 days.
    :param time_gap: The consecutive time interval/gap for each time range. Ignores a.time.limit.
    The format is based on a subset of the ISO-8601 duration format
    :return: facet.range=manufacturedate_dt&f.manufacturedate_dt.facet.range.start=2006-02-11T15:26:37Z&f.
    manufacturedate_dt.facet.range.end=2006-02-14T15:26:37Z&f.manufacturedate_dt.facet.range.gap=+1DAY
    """
    start, end = parse_datetime_range(time_filter)

    key_range_start = "f.{0}.facet.range.start".format(field)
    key_range_end = "f.{0}.facet.range.end".format(field)
    key_range_gap = "f.{0}.facet.range.gap".format(field)
    key_range_mincount = "f.{0}.facet.mincount".format(field)

    if time_gap:
        gap = gap_to_sorl(time_gap)
    else:
        gap = compute_gap(start, end, time_limit)

    value_range_start = start.get("parsed_datetime")
    if start.get("is_common_era"):
        value_range_start = start.get("parsed_datetime").isoformat().replace("+00:00", "") + "Z"

    value_range_end = start.get("parsed_datetime")
    if end.get("is_common_era"):
        value_range_end = end.get("parsed_datetime").isoformat().replace("+00:00", "") + "Z"

    value_range_gap = gap

    params = {
        'facet.range': field,
        key_range_start: value_range_start,
        key_range_end: value_range_end,
        key_range_gap: value_range_gap,
        key_range_mincount: 1
    }

    return params