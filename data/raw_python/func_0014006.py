def attr_case_name(self, name):
        """Returns preserved case name for case insensitive value of name.
        
        Checks first within standard attributes. If not found there, checks
        attributes for higher order data structures. If not found, returns
        supplied name as it is available for use. Intended to be used to help
        ensure that the same case is applied to all repetitions of a given
        variable name.
        
        Parameters
        ----------
        name : str
            name of variable to get stored case form

        Returns
        -------
        str
            name in proper case
        """

        lower_name = name.lower()
        for i in self.attrs():
            if lower_name == i.lower():
                return i
        # check if attribute present in higher order structures
        for key in self.keys_nD():
            for i in self[key].children.attrs():
                if lower_name == i.lower():
                    return i
        # nothing was found if still here
        # pass name back, free to be whatever
        return name