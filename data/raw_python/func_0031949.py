def close_connection(self):
        """
        Close connection kept by :meth:`connection`.

        If commit is needed, :meth:`sqlite3.Connection.commit`
        is called first and then :meth:`sqlite3.Connection.interrupt`
        is called.

        A few methods/generators support :meth:`close_connection`:

        - :meth:`search_command_record`
        - :meth:`select_by_command_record`

        """
        if self._db:
            db = self._db
            try:
                if self._need_commit:
                    db.commit()
            finally:
                db.interrupt()
                self._db = None
                self._need_commit = False