def _db_filename_from_dataframe(base_filename, df):
    """
    Generate database filename for a sqlite3 database we're going to
    fill with the contents of a DataFrame, using the DataFrame's
    column names and types.
    """
    db_filename = base_filename + ("_nrows%d" % len(df))
    for column_name in df.columns:
        column_db_type = db_type(df[column_name].dtype)
        column_name = column_name.replace(" ", "_")
        db_filename += ".%s_%s" % (column_name, column_db_type)
    return db_filename + ".db"