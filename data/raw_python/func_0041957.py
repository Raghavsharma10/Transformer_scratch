def set_auto_commit(self, auto_commit):
        """Sets auto-commit mode for this connection.

        If a connection is in auto-commit mode, then all its SQL statements will be executed and committed as individual
        transactions. Otherwise, its SQL statements are grouped into transactions that are terminated by a call to
        either the method commit or the method rollback. By default, new connections are in auto-commit mode.

        NOTE: If this method is called during a transaction and the auto-commit mode is changed, the transaction is
        committed. If this method is called and the auto-commit mode is not changed, the call is a no-op.

        :param auto_commit: `True` to enable auto-commit mode; `False` to disable it
        """
        auto_commit = bool(auto_commit)
        if auto_commit == self._auto_commit:
            return

        self._auto_commit = auto_commit

        if self.is_connected() and self._transaction_nesting_level != 0:
            self.commit_all()