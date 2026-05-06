def _open(self):
        """
        open input file, optionally with decompression
        """
        if self.fileName.endswith(".gz"):
            return gzip.open(self.fileName)
        elif self.fileName.endswith(".bz2"):
            return bz2.BZ2File(self.fileName)
        else:
            return open(self.fileName)