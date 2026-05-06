def row_csv_limiter(rows, limits=None):
    """
    Limit row passing a value or detect limits making the best effort.
    """

    limits = [None, None] if limits is None else limits

    if len(exclude_empty_values(limits)) == 2:
        upper_limit = limits[0]
        lower_limit = limits[1]
    elif len(exclude_empty_values(limits)) == 1:
        upper_limit = limits[0]
        lower_limit = row_iter_limiter(rows, 1, -1, 1)
    else:
        upper_limit = row_iter_limiter(rows, 0, 1, 0)
        lower_limit = row_iter_limiter(rows, 1, -1, 1)

    return rows[upper_limit: lower_limit]