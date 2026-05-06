def dict(self):
        """
        Return a deepcopy of self as a dictionary.

        All members that are ``Section`` instances are recursively turned to
        ordinary dictionaries - by calling their ``dict`` method.

        >>> n = a.dict()  # doctest: +SKIP
        >>> n == a  # doctest: +SKIP
        1
        >>> n is a  # doctest: +SKIP
        0
        """
        newdict = {}
        for entry in self:
            this_entry = self[entry]
            if isinstance(this_entry, Section):
                this_entry = this_entry.dict()
            elif isinstance(this_entry, list):
                # create a copy rather than a reference
                this_entry = list(this_entry)
            elif isinstance(this_entry, tuple):
                # create a copy rather than a reference
                this_entry = tuple(this_entry)
            newdict[entry] = this_entry
        return newdict