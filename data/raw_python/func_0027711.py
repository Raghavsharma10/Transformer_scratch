def checkpoint(self):
        """
        Update the database to reflect in-memory changes made to this item; for
        example, to make it show up in store.query() calls where it is now
        valid, but was not the last time it was persisted to the database.

        This is called automatically when in 'autocommit mode' (i.e. not in a
        transaction) and at the end of each transaction for every object that
        has been changed.
        """

        if self.store is None:
            raise NotInStore("You can't checkpoint %r: not in a store" % (self,))


        if self.__deleting:
            if not self.__everInserted:
                # don't issue duplicate SQL and crap; we were created, then
                # destroyed immediately.
                return
            self.store.executeSQL(self._baseDeleteSQL(self.store), [self.storeID])
            # re-using OIDs plays havoc with the cache, and with other things
            # as well.  We need to make sure that we leave a placeholder row at
            # the end of the table.
            if self.__deletingObject:
                # Mark this object as dead.
                self.store.executeSchemaSQL(_schema.CHANGE_TYPE,
                                            [-1, self.storeID])

                # Can't do this any more:
                # self.store.executeSchemaSQL(_schema.DELETE_OBJECT, [self.storeID])

                # TODO: need to measure the performance impact of this, then do
                # it to make sure things are in fact deleted:
                # self.store.executeSchemaSQL(_schema.APP_VACUUM)

            else:
                assert self.__legacy__

            # we're done...
            if self.store.autocommit:
                self.committed()
            return

        if self.__everInserted:
            # case 1: we've been inserted before, either previously in this
            # transaction or we were loaded from the db
            if not self.__dirty__:
                # we might have been checkpointed twice within the same
                # transaction; just don't do anything.
                return
            self.store.executeSQL(*self._updateSQL())
        else:
            # case 2: we are in the middle of creating the object, we've never
            # been inserted into the db before
            schemaAttrs = self.getSchema()

            insertArgs = [self.storeID]
            for (ignoredName, attrObj) in schemaAttrs:
                attrObjDuplicate, attributeValue = self.__dirty__[attrObj.attrname]
                # assert attrObjDuplicate is attrObj
                insertArgs.append(attributeValue)

            # XXX this isn't atomic, gross.
            self.store.executeSQL(self._baseInsertSQL(self.store), insertArgs)
            self.__everInserted = True
        # In case 1, we're dirty but we did an update, synchronizing the
        # database, in case 2, we haven't been created but we issue an insert.
        # In either case, the code in attributes.py sets the attribute *as well
        # as* populating __dirty__, so we clear out dirty and we keep the same
        # value, knowing it's the same as what's in the db.
        self.__dirty__.clear()
        if self.store.autocommit:
            self.committed()