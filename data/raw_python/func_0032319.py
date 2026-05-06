def db_from_dataframes(
        db_filename,
        dataframes,
        primary_keys={},
        indices={},
        subdir=None,
        overwrite=False,
        version=1):
    """
    Create a sqlite3 database from a collection of DataFrame objects

    Parameters
    ----------
    db_filename : str
        Name of database file to create

    dataframes : dict
        Dictionary from table names to DataFrame objects

    primary_keys : dict, optional
        Name of primary key column for each table

    indices : dict, optional
        Dictionary from table names to list of column name tuples

    subdir : str, optional

    overwrite : bool, optional
        If the database already exists, overwrite it?

    version : int, optional
    """
    if not (subdir is None or isinstance(subdir, str)):
        raise TypeError("Expected subdir to be None or str, got %s : %s" % (
            subdir, type(subdir)))
    db_path = build_path(db_filename, subdir)
    return db_from_dataframes_with_absolute_path(
        db_path,
        table_names_to_dataframes=dataframes,
        table_names_to_primary_keys=primary_keys,
        table_names_to_indices=indices,
        overwrite=overwrite,
        version=version)