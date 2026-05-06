def standard_db_name(file_column_name):
    """return a standard name by following rules:
    1. find all regular expression partners ((IDs)|(ID)|([A-Z][a-z]+)|([A-Z]{2,}))
    2. lower very part and join again with _
    This method is only used if values in table[model]['columns'] are str

    :param str file_column_name: name of column in file
    :return: standard name
    :rtype: str
    """
    found = id_re.findall(file_column_name)

    if not found:
        return file_column_name

    return '_'.join(x[0].lower() for x in found)