def get_var(self, name, recurse=True):
        """Return the first var of name ``name`` in the current
        scope stack (remember, vars are the ones that parse the
        input stream)

        :name: The name of the id
        :recurse: Whether parent scopes should also be searched (defaults to True)
        :returns: TODO

        """
        self._dlog("getting var '{}'".format(name))
        return self._search("vars", name, recurse)