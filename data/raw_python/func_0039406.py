def get_col_info(table_name, col_name, meta_file):
    """Return the content and metadata of a fiven column.

    Args:
        table_name(str): Name of the table.
        col_name(str): Name of the column.
        meta_file(str): Path to the meta.json file.

    Returns:
        tuple(pandas.Series, dict)
    """

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    data_table, table = load_data_table(table_name, meta_file, meta)

    for field in table['fields']:
        if field['name'] == col_name:
            col_meta = field

    col = data_table[col_name]

    return (col, col_meta)