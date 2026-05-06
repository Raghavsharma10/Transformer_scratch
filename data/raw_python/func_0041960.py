def set_nest_transactions_with_savepoints(self, nest_transactions_with_savepoints):
        """Sets if nested transactions should use savepoints.

        :param nest_transactions_with_savepoints: `True` or `False`
        """
        if self._transaction_nesting_level > 0:
            raise DBALConnectionError.may_not_alter_nested_transaction_with_savepoints_in_transaction()
        if not self._platform.is_savepoints_supported():
            raise DBALConnectionError.savepoints_not_supported()
        self._nest_transactions_with_savepoints = bool(nest_transactions_with_savepoints)