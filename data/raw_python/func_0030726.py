def _migration_required(connection):
    """ Returns True if ambry models do not match to db tables. Otherwise returns False. """
    stored_version = get_stored_version(connection)
    actual_version = SCHEMA_VERSION
    assert isinstance(stored_version, int)
    assert isinstance(actual_version, int)
    assert stored_version <= actual_version, \
        'Db version can not be greater than models version. Update your source code.'
    return stored_version < actual_version