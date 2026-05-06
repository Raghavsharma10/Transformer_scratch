def wipe(self):
        """
        Wipe the bolt database.

        Calling this after HoverPy has been instantiated is
        potentially dangerous. This function is mostly used
        internally for unit tests.
        """
        try:
            if os.isfile(self._dbpath):
                os.remove(self._dbpath)
        except OSError:
            pass