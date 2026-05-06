def jx_type(column):
    """
    return the jx_type for given column
    """
    if column.es_column.endswith(EXISTS_TYPE):
        return EXISTS
    return es_type_to_json_type[column.es_type]