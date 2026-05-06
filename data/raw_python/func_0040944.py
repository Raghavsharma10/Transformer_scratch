def get_csv_col_headers(rows, row_headers_count_value=0):
    """
    Retrieve csv column headers
    """
    count = 0

    if rows:
        for row in rows:
            if exclude_empty_values(row[:row_headers_count_value]):
                break
            count += 1

    if len(rows) == count:
        count = 1  # by default

    return [r[row_headers_count_value:] for r in rows[:count]]