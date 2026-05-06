def identifySQLError(self, sql, args, e):
        """
        Identify an appropriate SQL error object for the given message for the
        supported versions of sqlite.

        @return: an SQLError
        """
        message = e.args[0]
        if message.startswith("table") and message.endswith("already exists"):
            return errors.TableAlreadyExists(sql, args, e)
        return errors.SQLError(sql, args, e)