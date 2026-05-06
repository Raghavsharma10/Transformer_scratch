def leave_transaction_management(self) -> None:
        """
        End a transaction. Must not be dirty when doing so. ie. commit() or
        rollback() must be called if changes made. If dirty, changes will be
        discarded.
        """
        if len(self._transactions) == 0:
            raise RuntimeError("leave_transaction_management called outside transaction")
        elif len(self._transactions[-1]) > 0:
            raise RuntimeError("leave_transaction_management called with uncommited rollbacks")
        else:
            self._transactions.pop()