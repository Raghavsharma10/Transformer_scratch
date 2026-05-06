def populate_headers(headers):
    """
    Concatenate headers with subheaders
    """
    result = [''] * len(headers[0])
    values = [''] * len(headers)
    for k_index in range(0, len(headers)):
        for i_index in range(0, len(headers[k_index])):
            if headers[k_index][i_index]:
                values[k_index] = normalizer(
                    str(headers[k_index][i_index]))  # pass to str

            if len(exclude_empty_values(result)) > i_index:
                result[i_index] += "-{}".format(values[k_index])
            else:
                result[i_index] += str(values[k_index])

    return result