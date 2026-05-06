def db_from_dataframes_with_absolute_path(
        db_path,
        table_names_to_dataframes,
        table_names_to_primary_keys={},
        table_names_to_indices={},
        overwrite=False,
        version=1):
    """
    Create a sqlite3 database from a collection of DataFrame objects

    Parameters
    ----------
    db_path : str
        Path to database file to create

    table_names_to_dataframes : dict
        Dictionary from table names to DataFrame objects

    table_names_to_primary_keys : dict, optional
        Name of primary key column for each table

    table_names_to_indices : dict, optional
        Dictionary from table names to list of column name tuples

    overwrite : bool, optional
        If the database already exists, overwrite it?

    version : int, optional
    """
    if overwrite and exists(db_path):
        remove(db_path)

    tables = build_tables(
        table_names_to_dataframes,
        table_names_to_primary_keys,
        table_names_to_indices)
    return _create_cached_db(
        db_path,
        tables=tables,
        version=version)