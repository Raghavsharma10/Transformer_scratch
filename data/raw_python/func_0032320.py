def db_from_dataframe(
        db_filename,
        table_name,
        df,
        primary_key=None,
        subdir=None,
        overwrite=False,
        indices=(),
        version=1):
    """
    Given a dataframe `df`, turn it into a sqlite3 database.
    Store values in a table called `table_name`.

    Returns full path to the sqlite database file.
    """
    return db_from_dataframes(
        db_filename=db_filename,
        dataframes={table_name: df},
        primary_keys={table_name: primary_key},
        indices={table_name: indices},
        subdir=subdir,
        overwrite=overwrite,
        version=version)