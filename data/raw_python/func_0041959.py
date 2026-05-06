def get_transaction_isolation(self):
        """Returns the currently active transaction isolation level.

        :return: the current transaction isolation level
        :rtype: int
        """
        if self._transaction_isolation_level is None:
            self._transaction_isolation_level = self._platform.get_default_transaction_isolation_level()
        return self._transaction_isolation_level