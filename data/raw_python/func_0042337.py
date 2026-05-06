def tuple(data, field_name):
    """
    RETURN LIST  OF TUPLES
    """
    if isinstance(data, Cube):
        Log.error("not supported yet")

    if isinstance(data, FlatList):
        Log.error("not supported yet")

    if is_data(field_name) and "value" in field_name:
        # SIMPLIFY {"value":value} AS STRING
        field_name = field_name["value"]

    # SIMPLE PYTHON ITERABLE ASSUMED
    if is_text(field_name):
        if len(split_field(field_name)) == 1:
            return [(d[field_name],) for d in data]
        else:
            path = split_field(field_name)
            output = []
            flat_list._tuple1(data, path, 0, output)
            return output
    elif is_list(field_name):
        paths = [_select_a_field(f) for f in field_name]
        output = FlatList()
        _tuple((), unwrap(data), paths, 0, output)
        return output
    else:
        paths = [_select_a_field(field_name)]
        output = FlatList()
        _tuple((), data, paths, 0, output)
        return output