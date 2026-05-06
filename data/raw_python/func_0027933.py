def loaded(self, oself, dbval):
        """
        This method is invoked when the item is loaded from the database, and
        when a transaction is reverted which restores this attribute's value.

        @param oself: an instance of an item which has this attribute.

        @param dbval: the underlying database value which was retrieved.
        """
        setattr(oself, self.dbunderlying, dbval)
        delattr(oself, self.underlying)