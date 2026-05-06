def add(self, kind, key, *values):
        """Add processor functions.

        Any previous list of processors for `kind` and `key` will be
        overwritten.

        Parameters
        ----------
        kind : {"pre", "post"}
        key : str
            A registered key.  Add the functions (in order) to this key's list
            of processors.
        *values : callables
            Processors to add.
        """
        if kind == "pre":
            procs = self.pre
        elif kind == "post":
            procs = self.post
        else:
            raise ValueError("kind is not 'pre' or 'post'")
        self._check_if_registered(key)
        procs[key] = values