def build_tables(
        table_names_to_dataframes,
        table_names_to_primary_keys={},
        table_names_to_indices={}):
    """
    Parameters
    ----------
    table_names_to_dataframes : dict
        Dictionary mapping each table name to a DataFrame

    table_names_to_primary_keys : dict
        Dictionary mapping each table to its primary key

    table_names_to_indices : dict
        Dictionary mapping each table to a set of indices

    Returns list of DatabaseTable objects
    """
    tables = []
    for table_name, df in table_names_to_dataframes.items():
        table_indices = table_names_to_indices.get(table_name, [])
        primary_key = table_names_to_primary_keys.get(table_name)
        table = DatabaseTable.from_dataframe(
            name=table_name,
            df=df,
            indices=table_indices,
            primary_key=primary_key)
        tables.append(table)
    return tables