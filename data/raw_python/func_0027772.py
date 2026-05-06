def changed(self, item):
        """
        An item in this store was changed.  Add it to the current transaction's
        list of changed items, if a transaction is currently underway, or raise
        an exception if this L{Store} is currently in a state which does not
        allow changes.
        """
        if self._rejectChanges:
            raise errors.ChangeRejected()
        if self.transaction is not None:
            self.transaction.add(item)
            self.touched.add(item)