def getFileHandle(self, dataFile, openMethod):
        """
        Returns handle associated to the filename. If the file is
        already opened, update its priority in the cache and return
        its handle. Otherwise, open the file using openMethod, store
        it in the cache and return the corresponding handle.
        """
        if dataFile in self._memoTable:
            handle = self._memoTable[dataFile]
            self._update(dataFile, handle)
            return handle
        else:
            try:
                handle = openMethod(dataFile)
            except ValueError:
                raise exceptions.FileOpenFailedException(dataFile)

            self._memoTable[dataFile] = handle
            self._add(dataFile, handle)
            if len(self._memoTable) > self._maxCacheSize:
                dataFile = self._removeLru()
                del self._memoTable[dataFile]
            return handle