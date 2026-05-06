def rename(python_data: LdapObject, new_base_dn: str = None,
           database: Optional[Database] = None, **kwargs) -> LdapObject:
    """ Move/rename a LdapObject in the database. """
    table = type(python_data)
    dn = python_data.get_as_single('dn')
    assert dn is not None

    database = get_database(database)
    connection = database.connection

    # extract key and value from kwargs
    if len(kwargs) == 1:
        name, value = list(kwargs.items())[0]

        # work out the new rdn of the object
        split_new_rdn = [[(name, value, 1)]]

        field = _get_field_by_name(table, name)
        assert field.db_field

        python_data = python_data.merge({
            name: value,
        })

    elif len(kwargs) == 0:
        split_new_rdn = [str2dn(dn)[0]]
    else:
        assert False

    new_rdn = dn2str(split_new_rdn)

    connection.rename(
        dn,
        new_rdn,
        new_base_dn,
    )

    if new_base_dn is not None:
        split_base_dn = str2dn(new_base_dn)
    else:
        split_base_dn = str2dn(dn)[1:]

    tmp_list = [split_new_rdn[0]]
    tmp_list.extend(split_base_dn)

    new_dn = dn2str(tmp_list)

    python_data = python_data.merge({
        'dn': new_dn,
    })
    return python_data