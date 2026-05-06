def search(table: LdapObjectClass, query: Optional[Q] = None,
           database: Optional[Database] = None, base_dn: Optional[str] = None) -> Iterator[LdapObject]:
    """ Search for a object of given type in the database. """
    fields = table.get_fields()
    db_fields = {
        name: field
        for name, field in fields.items()
        if field.db_field
    }

    database = get_database(database)
    connection = database.connection

    search_options = table.get_search_options(database)

    iterator = tldap.query.search(
        connection=connection,
        query=query,
        fields=db_fields,
        base_dn=base_dn or search_options.base_dn,
        object_classes=search_options.object_class,
        pk=search_options.pk_field,
    )

    for dn, data in iterator:
        python_data = _db_to_python(data, table, dn)
        python_data = table.on_load(python_data, database)
        yield python_data