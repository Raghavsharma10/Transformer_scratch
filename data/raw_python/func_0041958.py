def set_transaction_isolation(self, level):
        """Sets the transaction isolation level.

        :param level: the level to set
        """
        self.ensure_connected()
        self._transaction_isolation_level = level
        self._platform.set_transaction_isolation(level)