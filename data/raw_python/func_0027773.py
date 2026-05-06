def transact(self, f, *a, **k):
        """
        Execute C{f(*a, **k)} in the context of a database transaction.

        Any changes made to this L{Store} by C{f} will be committed when C{f}
        returns.  If C{f} raises an exception, those changes will be reverted
        instead.

        If a transaction is already in progress (in this thread - ie, if a
        frame executing L{Store.transact} is already on the call stack), this
        will B{not} start a nested transaction.  Changes will not be committed
        until the existing transaction completes, and an exception raised by
        C{f} will not revert changes made by C{f}.  You probably don't want to
        ever call this if another transaction is in progress.

        @return: Whatever C{f(*a, **kw)} returns.
        @raise: Whatever C{f(*a, **kw)} raises, or a database exception.
        """
        if self.transaction is not None:
            return f(*a, **k)
        if self.attachedToParent:
            return self.parent.transact(f, *a, **k)
        try:
            self._begin()
            try:
                result = f(*a, **k)
                self.checkpoint()
            except:
                exc = Failure()
                try:
                    self.revert()
                except:
                    log.err(exc)
                    raise
                raise
            else:
                self._commit()
            return result
        finally:
            self._cleanupTxnState()