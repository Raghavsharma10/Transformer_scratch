def connection(connection, without_connection_test):
    """Set MySQL/MariaDB connection"""
    database.set_connection(connection=connection)

    if not without_connection_test:
        test_connection(connection)