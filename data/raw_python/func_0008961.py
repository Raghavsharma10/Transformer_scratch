def replace(self, target):
        """
        Rename this path to the given path, clobbering the existing
        destination if it exists.
        """
        if sys.version_info < (3, 3):
            raise NotImplementedError("replace() is only available "
                                      "with Python 3.3 and later")
        self._accessor.replace(self, target)