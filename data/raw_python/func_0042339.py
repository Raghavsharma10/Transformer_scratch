def select(data, field_name):
    """
    return list with values from field_name
    """
    if isinstance(data, Cube):
        return data._select(_normalize_selects(field_name))

    if isinstance(data, PartFlatList):
        return data.select(field_name)

    if isinstance(data, UniqueIndex):
        data = (
            data._data.values()
        )  # THE SELECT ROUTINE REQUIRES dicts, NOT Data WHILE ITERATING

    if is_data(data):
        return select_one(data, field_name)

    if is_data(field_name):
        field_name = wrap(field_name)
        if field_name.value in ["*", "."]:
            return data

        if field_name.value:
            # SIMPLIFY {"value":value} AS STRING
            field_name = field_name.value

    # SIMPLE PYTHON ITERABLE ASSUMED
    if is_text(field_name):
        path = split_field(field_name)
        if len(path) == 1:
            return FlatList([d[field_name] for d in data])
        else:
            output = FlatList()
            flat_list._select1(data, path, 0, output)
            return output
    elif is_list(field_name):
        keys = [_select_a_field(wrap(f)) for f in field_name]
        return _select(Data(), unwrap(data), keys, 0)
    else:
        keys = [_select_a_field(field_name)]
        return _select(Data(), unwrap(data), keys, 0)