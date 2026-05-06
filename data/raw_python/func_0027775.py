def getTableName(self, tableClass):
        """
        Retrieve the fully qualified name of the table holding items
        of a particular class in this store.  If the table does not
        exist in the database, it will be created as a side-effect.

        @param tableClass: an Item subclass

        @raises axiom.errors.ItemClassesOnly: if an object other than a
            subclass of Item is passed.

        @return: a string
        """
        if not (isinstance(tableClass, type) and issubclass(tableClass, item.Item)):
            raise errors.ItemClassesOnly("Only subclasses of Item have table names.")

        if tableClass not in self.typeToTableNameCache:
            self.typeToTableNameCache[tableClass] = self._tableNameFor(tableClass.typeName, tableClass.schemaVersion)
            # make sure the table exists
            self.getTypeID(tableClass)
        return self.typeToTableNameCache[tableClass]