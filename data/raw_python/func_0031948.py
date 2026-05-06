def connection(self, commit=False):
        """
        Context manager to keep around DB connection.

        :rtype: sqlite3.Connection

        SOMEDAY: Get rid of this function.  Keeping connection around as
        an argument to the method using this context manager is
        probably better as it is more explicit.
        Also, holding "global state" as instance attribute is bad for
        supporting threaded search, which is required for more fluent
        percol integration.

        """
        if commit:
            self._need_commit = True
        if self._db:
            yield self._db
        else:
            try:
                with self._get_db() as db:
                    self._db = db
                    db.create_function("REGEXP", 2, sql_regexp_func)
                    db.create_function("PROGRAM_NAME", 1,
                                       sql_program_name_func)
                    db.create_function("PATHDIST", 2, sql_pathdist_func)
                    yield self._db
                    if self._need_commit:
                        db.commit()
            finally:
                self._db = None
                self._need_commit = False