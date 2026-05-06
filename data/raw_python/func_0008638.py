def contents(self):
        """The raw file contents as a string."""
        if not self._contents:
            if self._path:
                # Read file into memory so we don't run out of file descriptors
                f = open(self._path, "rb")
                self._contents = f.read()
                f.close()
        return self._contents