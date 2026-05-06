def from_file(self, path=None):
        """Look for a version in ``self.version_file``, or in the specified
           path if supplied.
        """
        if self._version is None:
            self._version = file_version(path or self.version_file)
        return self