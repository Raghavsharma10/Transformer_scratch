def save(unique_keys, data, table_name='swdata'):
    """
    Save the given data to the table specified by `table_name`
    (which defaults to 'swdata'). The data must be a mapping
    or an iterable of mappings. Unique keys is a list of keys that exist
    for all rows and for which a unique index will be created.
    """

    _set_table(table_name)

    connection = _State.connection()

    if isinstance(data, Mapping):
        # Is a single datum
        data = [data]
    elif not isinstance(data, Iterable):
        raise TypeError("Data must be a single mapping or an iterable "
                        "of mappings")

    insert = _State.table.insert(prefixes=['OR REPLACE'])
    for row in data:
        if not isinstance(row, Mapping):
            raise TypeError("Elements of data must be mappings, got {}".format(
                            type(row)))
        fit_row(connection, row, unique_keys)
        connection.execute(insert.values(row))
    _State.check_last_committed()