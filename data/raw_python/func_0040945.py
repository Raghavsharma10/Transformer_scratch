def populate_csv_headers(rows,
                         partial_headers,
                         column_headers_count=1):
    """
    Populate csv rows headers when are empty, extending the superior or
    upper headers.
    """

    result = [''] * (len(rows) - column_headers_count)

    for i_index in range(0, len(partial_headers)):
        for k_index in range(0, len(partial_headers[i_index])):

            # missing field find for a value in upper rows
            if not partial_headers[i_index][k_index] and i_index - 1 >= 0:

                # TODO: It's necesary a for or only taking the
                # inmediate latest row works well??
                for t_index in range(i_index - 1, -1, -1):
                    # TODO: could suposse that allways a value exists
                    partial_value = partial_headers[t_index][k_index]
                    if partial_value:
                        partial_headers[i_index][k_index] = partial_value
                        break

        result[i_index] = " ".join(map(str, partial_headers[i_index]))

    return result