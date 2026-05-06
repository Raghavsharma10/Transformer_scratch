def getTypeID(self, tableClass):
        """
        Retrieve the typeID associated with a particular table in the
        in-database schema for this Store.  A typeID is an opaque integer
        representing the Item subclass, and the associated table in this
        Store's SQLite database.

        @param tableClass: a subclass of Item

        @return: an integer
        """
        key = (tableClass.typeName,
               tableClass.schemaVersion)
        if key in self.typenameAndVersionToID:
            return self.typenameAndVersionToID[key]
        return self.transact(self._maybeCreateTable, tableClass, key)