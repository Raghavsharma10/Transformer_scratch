def asterisk_to_min_max(field, time_filter, search_engine_endpoint, actual_params=None):
    """
    traduce [* TO *] to something like [MIN-INDEXED-DATE TO MAX-INDEXED-DATE]
    :param field: map the stats to this field.
    :param time_filter: this is the value to be translated. think in "[* TO 2000]"
    :param search_engine_endpoint: solr core
    :param actual_params: (not implemented) to merge with other params.
    :return: translated time filter
    """

    if actual_params:
        raise NotImplemented("actual_params")

    start, end = parse_solr_time_range_as_pair(time_filter)
    if start == '*' or end == '*':
        params_stats = {
            "q": "*:*",
            "rows": 0,
            "stats.field": field,
            "stats": "true",
            "wt": "json"
        }
        res_stats = requests.get(search_engine_endpoint, params=params_stats)

        if res_stats.ok:

            stats_date_field = res_stats.json()["stats"]["stats_fields"][field]
            date_min = stats_date_field["min"]
            date_max = stats_date_field["max"]

            if start != '*':
                date_min = start
            if end != '*':
                date_max = end

            time_filter = "[{0} TO {1}]".format(date_min, date_max)

    return time_filter