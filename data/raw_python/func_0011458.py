def get_local(self, name, recurse=True):
        """Get the local field (search for it) from the scope stack. An alias
        for ``get_var``

        :name: The name of the local field
        """
        self._dlog("getting local '{}'".format(name))
        return self._search("vars", name, recurse)