def show(self, *args, **kwds):
        """ Show how the SQL looks like when executed by the DB.

        This might not be supported by all connection types.
        For example: PostgreSQL does support it, SQLite does not.

        :rtype: str
        """
        # Same as in __call__, arguments win over keywords
        arg = args
        if not arg:
            arg = kwds  # pylint: disable=redefined-variable-type
        return self._db.show(self._sql, arg)