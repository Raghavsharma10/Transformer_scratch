def dirs(self, pattern=None):
        """ D.dirs() -> List of this directory's subdirectories.

        The elements of the list are Path objects.
        This does not walk recursively into subdirectories
        (but see :meth:`walkdirs`).

        With the optional `pattern` argument, this only lists
        directories whose names match the given pattern.  For
        example, ``d.dirs('build-*')``.
        """
        return [p for p in self.listdir(pattern) if p.isdir()]