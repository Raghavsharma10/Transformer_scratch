def filter(data, where):
    """
    where  - a function that accepts (record, rownum, rows) and returns boolean
    """
    if len(data) == 0 or where == None or where == TRUE:
        return data

    if isinstance(data, Container):
        return data.filter(where)

    if is_container(data):
        temp = jx_expression_to_function(where)
        dd = wrap(data)
        return wrap([unwrap(d) for i, d in enumerate(data) if temp(wrap(d), i, dd)])
    else:
        Log.error(
            "Do not know how to handle type {{type}}", type=data.__class__.__name__
        )

    try:
        return drill_filter(where, data)
    except Exception as _:
        # WOW!  THIS IS INEFFICIENT!
        return wrap(
            [unwrap(d) for d in drill_filter(where, [DataObject(d) for d in data])]
        )