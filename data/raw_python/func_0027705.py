def _currentlyValidAsReferentFor(self, store):
        """
        Is this object currently valid as a reference?  Objects which will be
        deleted in this transaction, or objects which are not in the same store
        are not valid.  See attributes.reference.__get__.
        """
        if store is None:
            # If your store is None, you can refer to whoever you want.  I'm in
            # a store but it doesn't matter that you're not.
            return True
        if self.store is not store:
            return False
        if self.__deletingObject:
            return False
        return True