def has_attr(self, name):
        """Returns boolean indicating presence of given attribute name
        
        Case-insensitive check
        
        Notes
        -----
        Does not check higher order meta objects
        
        Parameters
        ----------
        name : str
            name of variable to get stored case form
            
        Returns
        -------
        bool
            True if case-insesitive check for attribute name is True

        """

        if name.lower() in [i.lower() for i in self.data.columns]:
            return True
        return False