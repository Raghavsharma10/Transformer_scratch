def wiggleFileHandleToProtocol(self, fileHandle):
        """
        Return a continuous protocol object satsifiying the given query
        parameters from the given wiggle file handle.
        """
        for line in fileHandle:
            self.readWiggleLine(line)
        return self._data