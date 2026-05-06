def get_row_headers(rows, row_headers_count_value=0, column_headers_count=1):
    """
    Return row headers.
    Assume that by default it has one column header.
    Assume that there is only one father row header.
    """
    # TODO: REFACTOR ALGORITHM NEEDED
    partial_headers = []

    if row_headers_count_value:

        # Take partial data
        for k_index in range(0, len(rows) - column_headers_count):
            header = rows[k_index + column_headers_count][
                :row_headers_count_value]
            partial_headers.append(remove_list_duplicates(force_list(header)))

        # Populate headers
        populated_headers = populate_csv_headers(
            rows,
            partial_headers,
            column_headers_count)

        return populated_headers