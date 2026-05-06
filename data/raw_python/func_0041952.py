def begin_transaction(self):
        """Starts a transaction by suspending auto-commit mode."""
        self.ensure_connected()
        self._transaction_nesting_level += 1
        if self._transaction_nesting_level == 1:
            self._driver.begin_transaction()
        elif self._nest_transactions_with_savepoints:
            self.create_savepoint(self._get_nested_transaction_savepoint_name())