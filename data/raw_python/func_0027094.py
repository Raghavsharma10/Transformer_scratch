def commit(self):
        """
        Commit this transaction.
        """

        if not self._parent._is_active:
            raise exc.InvalidRequestError("This transaction is inactive")
        yield from self._do_commit()
        self._is_active = False