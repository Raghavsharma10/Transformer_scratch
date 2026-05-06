def _get_sqlite_columns(connection, table):
    """ Returns list of tuple containg columns of the table.

    Args:
        connection: sqlalchemy connection to sqlite database.
        table (str): name of the table

    Returns:
        list of (name, datatype, position): where name is column name, datatype is
            python type of the column, position is ordinal position of the column.

    """
    # TODO: Move to the sqlite wrapper.
    # TODO: Consider sqlalchemy mapping.
    SQL_TO_PYTHON_TYPES = {
        'INT': int,
        'INTEGER': int,
        'TINYINT': int,
        'SMALLINT': int,
        'MEDIUMINT': int,
        'BIGINT': int,
        'UNSIGNED BIG INT': int,
        'INT': int,
        'INT8': int,
        'NUMERIC': float,
        'REAL': float,
        'FLOAT': float,
        'DOUBLE': float,
        'BOOLEAN': bool,
        'CHARACTER': str,
        'VARCHAR': str,
        'TEXT': str
    }
    query = 'PRAGMA table_info(\'{}\');'
    result = connection.execute(query.format(table))
    ret = []

    for row in result:
        position = row[0] + 1
        name = row[1]
        datatype = row[2]
        try:
            datatype = SQL_TO_PYTHON_TYPES[datatype]
        except KeyError:
            raise Exception(
                'Do not know how to convert {} sql datatype to python data type.'
                .format(datatype))
        ret.append((name, datatype, position))
    return ret