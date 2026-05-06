def commit(self) -> None:
        """
        Attempt to commit all changes to LDAP database. i.e. forget all
        rollbacks.  However stay inside transaction management.
        """
        if len(self._transactions) == 0:
            raise RuntimeError("commit called outside transaction")

        # If we have nested transactions, we don't actually commit, but push
        # rollbacks up to previous transaction.
        if len(self._transactions) > 1:
            for on_rollback in reversed(self._transactions[-1]):
                self._transactions[-2].insert(0, on_rollback)

        _debug("commit")
        self.reset()