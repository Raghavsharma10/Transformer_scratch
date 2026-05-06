def window(data, param):
    """
    MAYBE WE CAN DO THIS WITH NUMPY (no, the edges of windows are not graceful with numpy)
    data - list of records
    """
    name = param.name  # column to assign window function result
    edges = param.edges  # columns to gourp by
    where = param.where  # DO NOT CONSIDER THESE VALUES
    sortColumns = param.sort  # columns to sort by
    calc_value = jx_expression_to_function(
        param.value
    )  # function that takes a record and returns a value (for aggregation)
    aggregate = param.aggregate  # WindowFunction to apply
    _range = (
        param.range
    )  # of form {"min":-10, "max":0} to specify the size and relative position of window

    data = filter(data, where)

    if not aggregate and not edges:
        if sortColumns:
            data = sort(data, sortColumns, already_normalized=True)
        # SIMPLE CALCULATED VALUE
        for rownum, r in enumerate(data):
            try:
                r[name] = calc_value(r, rownum, data)
            except Exception as e:
                raise e
        return

    try:
        edge_values = [e.value.var for e in edges]
    except Exception as e:
        raise Log.error("can only support simple variable edges", cause=e)

    if not aggregate or aggregate == "none":
        for _, values in groupby(data, edge_values):
            if not values:
                continue  # CAN DO NOTHING WITH THIS ZERO-SAMPLE

            if sortColumns:
                sequence = sort(values, sortColumns, already_normalized=True)
            else:
                sequence = values

            for rownum, r in enumerate(sequence):
                r[name] = calc_value(r, rownum, sequence)
        return

    for keys, values in groupby(data, edge_values):
        if not values:
            continue  # CAN DO NOTHING WITH THIS ZERO-SAMPLE

        sequence = sort(values, sortColumns)

        for rownum, r in enumerate(sequence):
            r["__temp__"] = calc_value(r, rownum, sequence)

        head = coalesce(_range.max, _range.stop)
        tail = coalesce(_range.min, _range.start)

        # PRELOAD total
        total = aggregate()
        for i in range(tail, head):
            total.add(sequence[i].__temp__)

        # WINDOW FUNCTION APPLICATION
        for i, r in enumerate(sequence):
            r[name] = total.end()
            total.add(sequence[i + head].__temp__)
            total.sub(sequence[i + tail].__temp__)

    for r in data:
        r["__temp__"] = None