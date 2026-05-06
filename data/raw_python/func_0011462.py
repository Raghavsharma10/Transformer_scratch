def get_id(self, name, recurse=True):
        """Get the first id matching ``name``. Will either be a local
        or a var.

        :name: TODO
        :returns: TODO

        """
        self._dlog("getting id '{}'".format(name))
        var = self._search("vars", name, recurse)
        return var