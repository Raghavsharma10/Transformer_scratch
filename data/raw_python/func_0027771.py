def batchInsert(self, itemType, itemAttributes, dataRows):
        """
        Create multiple items in the store without loading
        corresponding Python objects into memory.

        the items' C{stored} callback will not be called.

        Example::

            myData = [(37, u"Fred",  u"Wichita"),
                      (28, u"Jim",   u"Fresno"),
                      (43, u"Betty", u"Dubuque")]
            myStore.batchInsert(FooItem,
                                [FooItem.age, FooItem.name, FooItem.city],
                                myData)

        @param itemType: an Item subclass to create instances of.

        @param itemAttributes: an iterable of attributes on the Item subclass.

        @param dataRows: an iterable of iterables, each the same
        length as C{itemAttributes} and containing data corresponding
        to each attribute in it.

        @return: None.
        """
        class FakeItem:
            pass
        _NEEDS_DEFAULT = object() # token for lookup failure
        fakeOSelf = FakeItem()
        fakeOSelf.store = self
        sql = itemType._baseInsertSQL(self)
        indices = {}
        schema = [attr for (name, attr) in itemType.getSchema()]
        for i, attr in enumerate(itemAttributes):
            indices[attr] = i
        for row in dataRows:
            oid = self.store.executeSchemaSQL(
                _schema.CREATE_OBJECT, [self.store.getTypeID(itemType)])
            insertArgs = [oid]
            for attr in schema:
                i = indices.get(attr, _NEEDS_DEFAULT)
                if i is _NEEDS_DEFAULT:
                    pyval = attr.default
                else:
                    pyval = row[i]
                dbval = attr._convertPyval(fakeOSelf, pyval)
                insertArgs.append(dbval)
            self.executeSQL(sql, insertArgs)