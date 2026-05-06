def csv_format(csv_data, c_headers=None, r_headers=None, rows=None, **kwargs):
    """
    Format csv rows parsed to Dict or Array
    """
    result = None
    c_headers = [] if c_headers is None else c_headers
    r_headers = [] if r_headers is None else r_headers
    rows = [] if rows is None else rows

    result_format = kwargs.get('result_format', ARRAY_RAW_FORMAT)

    # DICT FORMAT
    if result_format == DICT_FORMAT:
        result = csv_dict_format(csv_data, c_headers, r_headers)

    # ARRAY_RAW_FORMAT
    elif result_format == ARRAY_RAW_FORMAT:
        result = rows

    # ARRAY_CLEAN_FORMAT
    elif result_format == ARRAY_CLEAN_FORMAT:
        result = csv_array_clean_format(csv_data, c_headers, r_headers)

    else:
        result = None

    # DEFAULT
    if result and result_format < DICT_FORMAT:
        result = [result]

    return result