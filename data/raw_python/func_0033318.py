def insert(self, data, **kwargs):
        """
        Calls the create method of OBJTYPE
        
        NOTE: this function is only properly usable on children classes that
        have overridden either OBJTYPE or PATH.
        
        @param data: the data of the new object to be created
        @param **kwargs: forwarded to create
        
        @raise DatabaseConflictError: if there is already an object with that
            ID_KEY and overwrite == False
        @raise MalformedObjectError: if a REQUIRED key of defaults is missing,
            or if the ID_KEY of the object is None and random_id is False
        """
        obj = self.OBJTYPE.create(data, path=self.PATH, **kwargs)
        self.append(obj)
        return obj