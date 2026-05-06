def save_account(changes: Changeset, table: LdapObjectClass, database: Database) -> Changeset:
    """ Modify a changes to add an automatically generated uidNumber. """
    d = {}
    settings = database.settings

    uid_number = changes.get_value_as_single('uidNumber')
    if uid_number is None:
        scheme = settings['NUMBER_SCHEME']
        first = settings.get('UID_FIRST', 10000)
        d['uidNumber'] = Counters.get_and_increment(
            scheme, "uidNumber", first,
            lambda n: not _check_exists(database, table, 'uidNumber', n)
        )

    changes = changes.merge(d)
    return changes