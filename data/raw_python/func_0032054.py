def preprocess_kwds(kwds):
    """
    Preprocess keyword arguments for `DataBase.search_command_record`.
    """
    from .utils.timeutils import parse_datetime, parse_duration

    for key in ['output', 'format', 'format_level',
                'with_command_id', 'with_session_id']:
        kwds.pop(key, None)

    for key in ['time_after', 'time_before']:
        val = kwds[key]
        if val:
            dt = parse_datetime(val)
            if dt:
                kwds[key] = dt

    for key in ['duration_longer_than', 'duration_less_than']:
        val = kwds[key]
        if val:
            dt = parse_duration(val)
            if dt:
                kwds[key] = dt

    # interpret "pattern" (currently just copying to --include-pattern)
    less_strict_pattern = list(map("*{0}*".format, kwds.pop('pattern', [])))
    kwds['match_pattern'] = kwds['match_pattern'] + less_strict_pattern

    if not kwds['sort_by']:
        kwds['sort_by'] = ['count']
    kwds['sort_by'] = [SORT_KEY_SYNONYMS[k] for k in kwds['sort_by']]
    return kwds