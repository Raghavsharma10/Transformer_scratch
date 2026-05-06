def deleteFromStore(self):
        """
        Delete all the Items which are found by this query.
        """
        if (self.limit is None and
            not isinstance(self.sort, attributes.UnspecifiedOrdering)):
            # The ORDER BY is pointless here, and SQLite complains about it.
            return self.cloneQuery(sort=None).deleteFromStore()

        #We can do this the fast way or the slow way.

        # If there's a 'deleted' callback on the Item type or 'deleteFromStore'
        # is overridden, we have to do it the slow way.
        deletedOverridden = (
            self.tableClass.deleted.im_func is not item.Item.deleted.im_func)
        deleteFromStoreOverridden = (
            self.tableClass.deleteFromStore.im_func is not
            item.Item.deleteFromStore.im_func)

        if deletedOverridden or deleteFromStoreOverridden:
            for it in self:
                it.deleteFromStore()
        else:

            # Find other item types whose instances need to be deleted
            # when items of the type in this query are deleted, and
            # remove them from the store.
            def itemsToDelete(attr):
                return attr.oneOf(self.getColumn("storeID"))

            if not item.allowDeletion(self.store, self.tableClass, itemsToDelete):
                raise errors.DeletionDisallowed(
                    'Cannot delete item; '
                    'has referents with whenDeleted == reference.DISALLOW')

            for it in item.dependentItems(self.store,
                                          self.tableClass, itemsToDelete):
                it.deleteFromStore()

            # actually run the DELETE for the items in this query.
            self._runQuery('DELETE', "")