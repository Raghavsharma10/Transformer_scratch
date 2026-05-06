def call_on_commit(self, callback):
    """Call a callback upon successful commit of a transaction.

    If not in a transaction, the callback is called immediately.

    In a transaction, multiple callbacks may be registered and will be
    called once the transaction commits, in the order in which they
    were registered.  If the transaction fails, the callbacks will not
    be called.

    If the callback raises an exception, it bubbles up normally.  This
    means: If the callback is called immediately, any exception it
    raises will bubble up immediately.  If the call is postponed until
    commit, remaining callbacks will be skipped and the exception will
    bubble up through the transaction() call.  (However, the
    transaction is already committed at that point.)
    """
    if not self.in_transaction():
      callback()
    else:
      self._on_commit_queue.append(callback)