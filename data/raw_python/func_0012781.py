def getextensibleindex(self, key, name):
        """
        Get the index of the first extensible item.

        Only for internal use. # TODO : hide this

        Parameters
        ----------
        key : str
            The type of IDF object. This must be in ALL_CAPS.
        name : str
            The name of the object to fetch.

        Returns
        -------
        int

        """
        return getextensibleindex(
            self.idfobjects, self.model, self.idd_info,
            key, name)