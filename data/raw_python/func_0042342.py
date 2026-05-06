def sort(data, fieldnames=None, already_normalized=False):
    """
    PASS A FIELD NAME, OR LIST OF FIELD NAMES, OR LIST OF STRUCTS WITH {"field":field_name, "sort":direction}
    """
    try:
        if data == None:
            return Null

        if not fieldnames:
            return wrap(sort_using_cmp(data, value_compare))

        if already_normalized:
            formal = fieldnames
        else:
            formal = query._normalize_sort(fieldnames)

        funcs = [(jx_expression_to_function(f.value), f.sort) for f in formal]

        def comparer(left, right):
            for func, sort_ in funcs:
                try:
                    result = value_compare(func(left), func(right), sort_)
                    if result != 0:
                        return result
                except Exception as e:
                    Log.error("problem with compare", e)
            return 0

        if is_list(data):
            output = FlatList([unwrap(d) for d in sort_using_cmp(data, cmp=comparer)])
        elif hasattr(data, "__iter__"):
            output = FlatList(
                [unwrap(d) for d in sort_using_cmp(list(data), cmp=comparer)]
            )
        else:
            Log.error("Do not know how to handle")
            output = None

        return output
    except Exception as e:
        Log.error("Problem sorting\n{{data}}", data=data, cause=e)