def get_type(self, name, recurse=True):
        """Get the names for the typename (created by typedef)

        :name: The typedef'd name to resolve
        :returns: An array of resolved names associated with the typedef'd name

        """
        self._dlog("getting type '{}'".format(name))
        return self._search("types", name, recurse)