def _db_to_python(db_data: dict, table: LdapObjectClass, dn: str) -> LdapObject:
    """ Convert a DbDate object to a LdapObject. """
    fields = table.get_fields()

    python_data = table({
        name: field.to_python(db_data[name])
        for name, field in fields.items()
        if field.db_field
    })
    python_data = python_data.merge({
        'dn': dn,
    })
    return python_data