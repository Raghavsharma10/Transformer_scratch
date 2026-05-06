def rollback(self):
        """Cancels any database changes done during the current transaction."""
        if self._transaction_nesting_level == 0:
            raise DBALConnectionError.no_active_transaction()

        self.ensure_connected()
        if self._transaction_nesting_level == 1:
            self._transaction_nesting_level = 0
            self._driver.rollback()
            self._is_rollback_only = False
            if not self._auto_commit:
                self.begin_transaction()
        elif self._nest_transactions_with_savepoints:
            self.rollback_savepoint(self._get_nested_transaction_savepoint_name())
            self._transaction_nesting_level -= 1
        else:
            self._is_rollback_only = True
            self._transaction_nesting_level -= 1