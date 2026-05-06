def save(changes: Changeset, database: Optional[Database] = None) -> LdapObject:
    """ Save all changes in a LdapChanges. """
    assert isinstance(changes, Changeset)

    if not changes.is_valid:
        raise RuntimeError(f"Changeset has errors {changes.errors}.")

    database = get_database(database)
    connection = database.connection

    table = type(changes._src)

    # Run hooks on changes
    changes = table.on_save(changes, database)

    # src dn   | changes dn | result         | action
    # ---------------------------------------|--------
    # None     | None       | error          | error
    # None     | provided   | use changes dn | create
    # provided | None       | use src dn     | modify
    # provided | provided   | error          | error

    src_dn = changes.src.get_as_single('dn')
    if src_dn is None and 'dn' not in changes:
        raise RuntimeError("No DN was given")
    elif src_dn is None and 'dn' in changes:
        dn = changes.get_value_as_single('dn')
        assert dn is not None
        create = True
    elif src_dn is not None and 'dn' not in changes:
        dn = src_dn
        assert dn is not None
        create = False
    else:
        raise RuntimeError("Changes to DN are not supported.")

    assert dn is not None

    if create:
        # Add new entry
        mod_list = _python_to_mod_new(changes)
        try:
            connection.add(dn, mod_list)
        except ldap3.core.exceptions.LDAPEntryAlreadyExistsResult:
            raise ObjectAlreadyExists(
                "Object with dn %r already exists doing add" % dn)
    else:
        mod_list = _python_to_mod_modify(changes)
        if len(mod_list) > 0:
            try:
                connection.modify(dn, mod_list)
            except ldap3.core.exceptions.LDAPNoSuchObjectResult:
                raise ObjectDoesNotExist(
                    "Object with dn %r doesn't already exist doing modify" % dn)

    # get new values
    python_data = table(changes.src.to_dict())
    python_data = python_data.merge(changes.to_dict())
    python_data = python_data.on_load(python_data, database)
    return python_data