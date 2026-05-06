def row_iter_limiter(rows, begin_row, way, c_value):
    """
    Alghoritm to detect row limits when row have more that one column.
    Depending the init params find from the begin or behind.
    NOT SURE THAT IT WORKS WELL..
    """
    limit = None

    for index in range(begin_row, len(rows)):
        if not len(exclude_empty_values(rows[way * index])) == 1:
            limit = way * index + c_value if way * index + \
                c_value not in [way * len(rows), 0] else None
            break

    return limit