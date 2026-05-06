def add_target(self, name=None):
        """
        Add an SCons target to this nest.

        The function decorated will be immediately called with each of the
        output directories and current control dictionaries. Each result will
        be added to the respective control dictionary for later nests to
        access.

        :param name: Name for the target in the name (default: function name).
        """
        def deco(func):
            def nestfunc(control):
                destdir = os.path.join(self.dest_dir, control['OUTDIR'])
                return [func(destdir, control)]
            key = name or func.__name__
            self.nest.add(key, nestfunc, create_dir=False)
            self._register_alias(key)
            return func
        return deco