def get_size(self):
        """
        Get size of this VideoFile in bytes
        :return: size as integer
        """
        if self._size is None:
            self._size = self._filepath.stat().st_size
        return self._size