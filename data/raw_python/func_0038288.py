def sql_column_type(column_data, prefer_varchar=False, prefer_int=False):
    """
    Retrieve the best fit data type for a column of a MySQL table.

    Accepts a iterable of values ONLY for the column whose data type
    is in question.

    :param column_data: Iterable of values from a MySQL table column
    :param prefer_varchar: Use type VARCHAR if valid
    :param prefer_int: Use type INT if valid
    :return: data type
    """
    # Collect list of type, length tuples
    type_len_pairs = [ValueType(record).get_type_len for record in column_data]

    # Retrieve frequency counts of each type
    types_count = {t: type_len_pairs.count(t) for t in set([type_ for type_, len_, len_dec in type_len_pairs])}

    # Most frequently occurring datatype
    most_frequent = max(types_count.items(), key=itemgetter(1))[0]

    # Get max length of all rows to determine suitable limit
    len_lst, len_decimals_lst = [], []
    for type_, len_, len_dec in type_len_pairs:
        if type_ == most_frequent:
            if type(len_) is int:
                len_lst.append(len_)
            if type(len_dec) is int:
                len_decimals_lst.append(len_dec)

    # Catch errors if current type has no len
    try:
        max_len = max(len_lst)
    except ValueError:
        max_len = None
    try:
        max_len_decimal = max(len_decimals_lst)
    except ValueError:
        max_len_decimal = None

    # Return VARCHAR or INT type if flag is on
    if prefer_varchar and most_frequent != 'VARCHAR' and 'text' in most_frequent.lower():
        most_frequent = 'VARCHAR'
    elif prefer_int and most_frequent != 'INT' and 'int' in most_frequent.lower():
        most_frequent = 'INT'

    # Return MySQL datatype in proper format, only include length if it is set
    if max_len and max_len_decimal:
        return '{0} ({1}, {2})'.format(most_frequent, max_len, max_len_decimal)
    elif max_len:
        return '{0} ({1})'.format(most_frequent, max_len)
    else:
        return most_frequent