def filterRead(self, read):
        """
        Filter a read, according to our set of filters.

        @param read: A C{Read} instance or one of its subclasses.
        @return: C{False} if the read fails any of our filters, else the
            C{Read} instance returned by our list of filters.
        """
        for filterFunc in self._filters:
            filteredRead = filterFunc(read)
            if filteredRead is False:
                return False
            else:
                read = filteredRead
        return read