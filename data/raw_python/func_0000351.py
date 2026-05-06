def insert(python_data: LdapObject, database: Optional[Database] = None) -> LdapObject:
    """ Insert a new python_data object in the database. """
    assert isinstance(python_data, LdapObject)

    table: LdapObjectClass = type(python_data)

    # ADD NEW ENTRY
    empty_data = table()
    changes = changeset(empty_data, python_data.to_dict())

    return save(changes, database)