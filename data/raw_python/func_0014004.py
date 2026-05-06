def var_case_name(self, name):
        """Provides stored name (case preserved) for case insensitive input
        
        If name is not found (case-insensitive check) then name is returned,
        as input. This function is intended to be used to help ensure the
        case of a given variable name is the same across the Meta object.
        
        Parameters
        ----------
        name : str
            variable name in any case
            
        Returns
        -------
        str
            string with case preserved as in metaobject
            
        """

        lower_name = name.lower()
        if name in self:
            for i in self.keys():
                if lower_name == i.lower():
                    return i
            for i in self.keys_nD():
                if lower_name == i.lower():
                    return i
        return name