def csv_to_dict(csv_filepath, **kwargs):
    """
    Turn csv into dict.
    Args:
        :csv_filepath: path to csv file to turn into dict.
        :limits: path to csv file to turn into dict
    """
    callbacks = {'to_list': csv_tolist,
                 'row_csv_limiter': row_csv_limiter,
                 'csv_row_cleaner': csv_row_cleaner,
                 'row_headers_count': row_headers_count,
                 'get_col_header': get_csv_col_headers,
                 'get_row_headers': get_row_headers,
                 'populate_headers': populate_headers,
                 'csv_column_header_cleaner': csv_column_header_cleaner,
                 'csv_column_cleaner': csv_column_cleaner,
                 'retrieve_csv_data': retrieve_csv_data}

    callbacks.update(kwargs.get('alt_callbacks', {}))
    rows = kwargs.get('rows', [])

    if not rows:
        # csv_tolist of rows
        rows = callbacks.get('to_list')(csv_filepath, **kwargs)

        if not rows:
            msg = 'Empty rows obtained from {}'.format(csv_filepath)
            logger.warning(msg)
            raise ValueError(msg)

    # apply limits
    rows = callbacks.get('row_csv_limiter')(
        rows, kwargs.get('limits', [None, None]))

    # apply row cleaner
    rows = callbacks.get('csv_row_cleaner')(rows)

    # apply column cleaner
    rows = callbacks.get('csv_column_cleaner')(rows)

    # count raw headers
    num_row_headers = callbacks.get('row_headers_count')(rows)

    # take colum_headers
    c_headers_raw = callbacks.get('get_col_header')(rows, num_row_headers)

    # get row_headers
    r_headers = callbacks.get('get_row_headers')(
        rows, num_row_headers, len(c_headers_raw))

    # format colum_headers
    c_headers_dirty = callbacks.get('populate_headers')(
        c_headers_raw) if len(c_headers_raw) > 1 else c_headers_raw[0]

    # Clean csv column headers of empty values.
    c_headers = callbacks.get('csv_column_header_cleaner')(c_headers_dirty)

    # take data
    csv_data = callbacks.get('retrieve_csv_data')(
        rows,
        column_header=len(c_headers_raw),
        row_header=num_row_headers,
        limit_column=len(c_headers) - len(c_headers_dirty) or None)

    # Check column headers validation
    if csv_data:
        assert len(c_headers) == len(csv_data[0])

    # Check row headers validation
    if r_headers:
        assert len(r_headers) == len(csv_data)

    # Transform rows into dict zipping the headers.
    kwargs.pop('rows', None)
    result = csv_format(csv_data, c_headers, r_headers, rows, **kwargs)

    return result