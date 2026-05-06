def files(self, pattern=None):
        """ D.files() -> List of the files in this directory.

        The elements of the list are Path objects.
        This does not walk into subdirectories (see :meth:`walkfiles`).

        With the optional `pattern` argument, this only lists files
        whose names match the given pattern.  For example,
        ``d.files('*.pyc')``.
        """

        return [p for p in self.listdir(pattern) if p.isfile()]