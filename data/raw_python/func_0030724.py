def get_stored_version(connection):
    """ Returns database version.

    Args:
        connection (sqlalchemy connection):

    Raises: Assuming user_version pragma (sqlite case) and user_version table (postgresql case)
        exist because they created with the database creation.

    Returns:
        int: version of the database.

    """

    if connection.engine.name == 'sqlite':
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version == 0:
            raise VersionIsNotStored
        return version
    elif connection.engine.name == 'postgresql':
        try:
            r = connection\
                .execute('SELECT version FROM {}.user_version;'.format(POSTGRES_SCHEMA_NAME))\
                .fetchone()
            if not r:
                raise VersionIsNotStored

            version = r[0]

        except ProgrammingError:
            # This happens when the user_version table doesn't exist
            raise VersionIsNotStored
        return version
    else:
        raise DatabaseError('Do not know how to get version from {} engine.'.format(connection.engine.name))