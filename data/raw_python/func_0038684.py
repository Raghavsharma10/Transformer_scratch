def filter_from_query(query, id_field="id", field_map={}):
    """This returns a filter which actually filters out everything, unlike the
    preview filter which includes excluded_ids for UI purposes.
    """
    f = groups_filter_from_query(query, field_map=field_map)
    excluded_ids = query.get("excluded_ids")
    included_ids = query.get("included_ids")

    if included_ids:  # include these, please
        if f is None:
            f = Terms(pk=included_ids)
        else:
            f |= Terms(pk=included_ids)

    if excluded_ids:  # exclude these
        if f is None:
            f = MatchAll()

        f &= ~Terms(pk=excluded_ids)
    return f