def csv_array_clean_format(csv_data, c_headers=None, r_headers=None):
    """
    Format csv rows parsed to Array clean format.
    """

    result = []
    real_num_header = len(force_list(r_headers[0])) if r_headers else 0
    result.append([""] * real_num_header + c_headers)

    for k_index in range(0, len(csv_data)):

        if r_headers:
            result.append(
                list(
                    itertools.chain(
                        [r_headers[k_index]],
                        csv_data[k_index])))

        else:
            result.append(csv_data[k_index])

    return result