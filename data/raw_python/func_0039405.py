def load_data_table(table_name, meta_file, meta):
    """Return the contents and metadata of a given table.

    Args:
        table_name(str): Name of the table.
        meta_file(str): Path to the meta.json file.
        meta(dict): Contents of meta.json.

    Returns:
        tuple(pandas.DataFrame, dict)

    """
    for table in meta['tables']:
        if table['name'] == table_name:
            prefix = os.path.dirname(meta_file)
            relative_path = os.path.join(prefix, meta['path'], table['path'])
            return pd.read_csv(relative_path), table