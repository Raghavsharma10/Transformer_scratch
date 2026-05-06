def removeextensibles(self, key, name):
        """
        Remove extensible items in the object of key and name.

        Only for internal use. # TODO : hide this

        Parameters
        ----------
        key : str
            The type of IDF object. This must be in ALL_CAPS.
        name : str
            The name of the object to fetch.

        Returns
        -------
        EpBunch object

        """
        return removeextensibles(
            self.idfobjects, self.model, self.idd_info,
            key, name)