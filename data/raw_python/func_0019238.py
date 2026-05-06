def pysourcefiles(self):
        """All source files of the actual models Python classes and their
        respective base classes."""
        sourcefiles = set()
        for (name, child) in vars(self).items():
            try:
                parents = inspect.getmro(child)
            except AttributeError:
                continue
            for parent in parents:
                try:
                    sourcefile = inspect.getfile(parent)
                except TypeError:
                    break
                sourcefiles.add(sourcefile)
        return Lines(*sourcefiles)