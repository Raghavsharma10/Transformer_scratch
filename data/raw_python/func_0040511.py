def to_dict_formatter(row, cursor):
    """ Take a row and use the column names from cursor to turn the row into a
    dictionary.

    Note: converts column names to lower-case!

    :param row: one database row, sequence of column values
    :type row: (value, ...)
    :param cursor: the cursor which was used to make the query
    :type cursor: DB-API cursor object
    """
    # Empty row? Return.
    if not row:
        return row
    # No cursor? Raise runtime error.
    if cursor is None or cursor.description is None:
        raise RuntimeError("No DB-API cursor or description available.")

    # Give each value the appropriate column name within in the resulting
    # dictionary.
    column_names = (d[0] for d in cursor.description)  # 0 is the name
    return {name: value for value, name in zip(row, column_names)}