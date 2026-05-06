def _writeRec(self, fh, rec):
        """
        Writes a single record to a file provided by the filehandle fh.
        """
        fh.write(str(rec) + "\n")
        for child in sorted(rec.children, key=self._recSortKey):
            self._writeRec(fh, child)