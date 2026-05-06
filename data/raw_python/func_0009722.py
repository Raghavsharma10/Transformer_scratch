def get_var(name, default=None):
    """
    Returns the variable with the provided key from the
    table specified by _State.vars_table_name.
    """
    alchemytypes = {"text": lambda x: x.decode('utf-8'),
                    "big_integer": lambda x: int(x),
                    "date": lambda x: x.decode('utf-8'),
                    "datetime": lambda x: x.decode('utf-8'),
                    "float": lambda x: float(x),
                    "large_binary": lambda x: x,
                    "boolean": lambda x: x==b'True'}

    connection = _State.connection()
    _State.new_transaction()

    if _State.vars_table_name not in list(_State.metadata.tables.keys()):
        return None

    table = sqlalchemy.Table(_State.vars_table_name, _State.metadata)
    s = sqlalchemy.select([table.c.value_blob, table.c.type])
    s = s.where(table.c.name == name)
    result = connection.execute(s).fetchone()

    if not result:
        return None

    return alchemytypes[result[1]](result[0])

    # This is to do the variable type conversion through the SQL engine
    execute = connection.execute
    execute("CREATE TEMPORARY TABLE _sw_tmp ('value' {})".format(result.type))
    execute("INSERT INTO _sw_tmp VALUES (:value)", value=result.value_blob)
    var = execute('SELECT value FROM _sw_tmp').fetchone().value
    execute("DROP TABLE _sw_tmp")
    return var.decode('utf-8')