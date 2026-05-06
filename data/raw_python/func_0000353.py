def delete(python_data: LdapObject, database: Optional[Database] = None) -> None:
    """ Delete a LdapObject from the database. """
    dn = python_data.get_as_single('dn')
    assert dn is not None

    database = get_database(database)
    connection = database.connection

    connection.delete(dn)