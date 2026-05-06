def get_condition_filter(condition, field_map={}):
    """
    Return the appropriate filter for a given group condition.

    #  TODO: integrate this into groups_filter_from_query function.
    """
    field_name = condition.get("field")
    field_name = field_map.get(field_name, field_name)
    operation = condition["type"]
    values = condition["values"]

    condition_filter = MatchAll()

    if values:
        values = [v["value"] for v in values]
        if operation == "all":
            for value in values:
                if "." in field_name:
                    path = field_name.split(".")[0]
                    condition_filter &= Nested(path=path, filter=Term(**{field_name: value}))
                else:
                    condition_filter &= Term(**{field_name: value})
        elif operation == "any":
            if "." in field_name:
                path = field_name.split(".")[0]
                condition_filter &= Nested(path=path, filter=Terms(**{field_name: values}))
            else:
                condition_filter &= Terms(**{field_name: values})
        elif operation == "none":
            if "." in field_name:
                path = field_name.split(".")[0]
                condition_filter &= ~Nested(path=path, filter=Terms(**{field_name: values}))
            else:
                condition_filter &= ~Terms(**{field_name: values})
        else:
            raise ValueError(
                """ES conditions must be one of the following values: ['all', 'any', 'none']"""
            )
    return condition_filter