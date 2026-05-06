def filenames(self):
        """A list of all handled auxiliary file names.

        >>> from hydpy import dummies
        >>> dummies.v2af.filenames
        ['file1', 'file2']
        """
        fns = set()
        for fn2var in self._type2filename2variable.values():
            fns.update(fn2var.keys())
        return sorted(fns)