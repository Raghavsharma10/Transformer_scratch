def listdir(self, pattern=None):
        """ D.listdir() -> List of items in this directory.

        Use :meth:`files` or :meth:`dirs` instead if you want a listing
        of just files or just subdirectories.

        The elements of the list are Path objects.

        With the optional `pattern` argument, this only lists
        items whose names match the given pattern.

        .. seealso:: :meth:`files`, :meth:`dirs`
        """
        if pattern is None:
            pattern = '*'
        return [
            self / child
            for child in map(self._always_unicode, os.listdir(self))
            if self._next_class(child).fnmatch(pattern)
        ]