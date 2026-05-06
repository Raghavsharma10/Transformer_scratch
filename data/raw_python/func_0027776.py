def getColumnName(self, attribute):
        """
        Retreive the fully qualified column name for a particular
        attribute in this store.  The attribute must be bound to an
        Item subclass (its type must be valid). If the underlying
        table does not exist in the database, it will be created as a
        side-effect.

        @param tableClass: an Item subclass

        @return: a string
        """
        if attribute not in self.attrToColumnNameCache:
            self.attrToColumnNameCache[attribute] = '.'.join(
                (self.getTableName(attribute.type),
                 self.getShortColumnName(attribute)))
        return self.attrToColumnNameCache[attribute]