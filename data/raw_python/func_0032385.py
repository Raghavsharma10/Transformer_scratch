def local_path(self, url, filename=None, decompress=False, download=False):
        """
        What will the full local path be if we download the given file?
        """
        if download:
            return self.fetch(url=url, filename=filename, decompress=decompress)
        else:
            filename = self.local_filename(url, filename, decompress)
            return join(self.cache_directory_path, filename)