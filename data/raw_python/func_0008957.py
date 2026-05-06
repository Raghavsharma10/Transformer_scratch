def iterdir(self):
        """Iterate over the files in this directory.  Does not yield any
        result for the special paths '.' and '..'.
        """
        for name in self._accessor.listdir(self):
            if name in ('.', '..'):
                # Yielding a path object for these makes little sense
                continue
            yield self._make_child_relpath(name)