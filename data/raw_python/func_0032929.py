def _set_autocommit(connection):
        """Make sure a connection is in autocommit mode."""
        if hasattr(connection.connection, "autocommit"):
            if callable(connection.connection.autocommit):
                connection.connection.autocommit(True)
            else:
                connection.connection.autocommit = True
        elif hasattr(connection.connection, "set_isolation_level"):
            connection.connection.set_isolation_level(0)