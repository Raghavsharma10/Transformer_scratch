def _convertPyval(self, oself, pyval):
        """
        Convert a Python value to a value suitable for inserting into the
        database.

        @param oself: The object on which this descriptor is an attribute.
        @param pyval: The value to be converted.
        @return: A value legal for this column in the database.
        """
        # convert to dbval later, I guess?
        if pyval is None and not self.allowNone:
            raise TypeError("attribute [%s.%s = %s()] must not be None" % (
                    self.classname, self.attrname, self.__class__.__name__))

        return self.infilter(pyval, oself, oself.store)