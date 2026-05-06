def exists(self, url, filename=None, decompress=False):
        """
        Return True if a local file corresponding to these arguments
        exists.
        """
        return download.file_exists(
            url,
            filename=filename,
            decompress=decompress,
            subdir=self.subdir)