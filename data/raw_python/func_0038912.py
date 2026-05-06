def checksum(self, path, hashtype='sha1'):
        """Returns the checksum of the given path."""
        return self._handler.checksum(hashtype, posix_path(path))