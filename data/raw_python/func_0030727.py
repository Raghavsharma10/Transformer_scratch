def _update_version(connection, version):
    """ Updates version in the db to the given version.

    Args:
        connection (sqlalchemy connection): sqlalchemy session where to update version.
        version (int): version of the migration.

    """
    if connection.engine.name == 'sqlite':
        connection.execute('PRAGMA user_version = {}'.format(version))

    elif connection.engine.name == 'postgresql':

        connection.execute(DDL('CREATE SCHEMA IF NOT EXISTS {};'.format(POSTGRES_SCHEMA_NAME)))
        connection.execute(DDL('CREATE SCHEMA IF NOT EXISTS {};'.format(POSTGRES_PARTITION_SCHEMA_NAME)))

        connection.execute('CREATE TABLE IF NOT EXISTS {}.user_version(version INTEGER NOT NULL);'
                           .format(POSTGRES_SCHEMA_NAME))

        # upsert.
        if connection.execute('SELECT * FROM {}.user_version;'.format(POSTGRES_SCHEMA_NAME)).fetchone():
            # update
            connection.execute('UPDATE {}.user_version SET version = {};'
                               .format(POSTGRES_SCHEMA_NAME, version))
        else:
            # insert
            connection.execute('INSERT INTO {}.user_version (version) VALUES ({})'
                               .format(POSTGRES_SCHEMA_NAME, version))
    else:
        raise DatabaseMissingError('Do not know how to migrate {} engine.'
                                   .format(connection.engine.driver))