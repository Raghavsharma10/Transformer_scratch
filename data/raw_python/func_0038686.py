def groups_filter_from_query(query, field_map={}):
    """Creates an F object for the groups of a search query."""
    f = None
    # filter groups
    for group in query.get("groups", []):
        group_f = MatchAll()
        for condition in group.get("conditions", []):
            field_name = condition["field"]
            field_name = field_map.get(field_name, field_name)
            operation = condition["type"]
            values = condition["values"]
            if values:
                values = [v["value"] for v in values]
                if operation == "all":
                    # NOTE: is there a better way to express this?
                    for value in values:
                        if "." in field_name:
                            path = field_name.split(".")[0]
                            group_f &= Nested(path=path, filter=Term(**{field_name: value}))
                        else:
                            group_f &= Term(**{field_name: value})
                elif operation == "any":
                    if "." in field_name:
                        path = field_name.split(".")[0]
                        group_f &= Nested(path=path, filter=Terms(**{field_name: values}))
                    else:
                        group_f &= Terms(**{field_name: values})
                elif operation == "none":
                    if "." in field_name:
                        path = field_name.split(".")[0]
                        group_f &= ~Nested(path=path, filter=Terms(**{field_name: values}))
                    else:
                        group_f &= ~Terms(**{field_name: values})

        date_range = group.get("time")
        if date_range:
            group_f &= date_range_filter(date_range)
        if f:
            f |= group_f
        else:
            f = group_f
    return f