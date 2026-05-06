def close(self):
        """
        Close this transaction.

        If this transaction is the base transaction in a begin/commit
        nesting, the transaction will rollback().  Otherwise, the
        method returns.

        This is used to cancel a Transaction without affecting the scope of
        an enclosing transaction.
        """
        if not self._connection or not self._parent:
            return
        if not self._parent._is_active:
            # pragma: no cover
            self._connection = None
            # self._parent = None
            return
        if self._parent is self:
            yield from self.rollback()
        else:
            self._is_active = False
        self._connection = None
        self._parent = None