def csv_dict_format(csv_data, c_headers=None, r_headers=None):
    """
    Format csv rows parsed to Dict.
    """
    # format dict if has row_headers
    if r_headers:
        result = {}
        for k_index in range(0, len(csv_data)):
            if r_headers[k_index]:
                result[r_headers[k_index]] = collections.OrderedDict(
                    zip(c_headers, csv_data[k_index]))

    # format list if hasn't row_headers -- square csv
    else:
        result = []
        for k_index in range(0, len(csv_data)):
            result.append(
                collections.OrderedDict(zip(c_headers, csv_data[k_index])))
        result = [result]

    return result