def preview_filter_from_query(query, id_field="id", field_map={}):
    """This filter includes the "excluded_ids" so they still show up in the editor."""
    f = groups_filter_from_query(query, field_map=field_map)
    # NOTE: we don't exclude the excluded ids here so they show up in the editor
    # include these, please
    included_ids = query.get("included_ids")
    if included_ids:
        if f:
            f |= Terms(pk=included_ids)
        else:
            f = Terms(pk=included_ids)
    return f