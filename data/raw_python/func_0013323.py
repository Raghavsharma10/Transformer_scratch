def write(self, fh):
        """
        Write set to a GFF3 format file.

        :param file fh: file handle for file to write to
        """
        fh.write(GFF3_HEADER+"\n")
        for root in sorted(self.roots, key=self._recSortKey):
            self._writeRec(fh, root)