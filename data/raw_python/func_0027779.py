def _maybeCreateTable(self, tableClass, key):
        """
        A type ID has been requested for an Item subclass whose table was not
        present when this Store was opened.  Attempt to create the table, and
        if that fails because another Store object (perhaps in another process)
        has created the table, re-read the schema.  When that's done, return
        the typeID.

        This method is internal to the implementation of getTypeID.  It must be
        run in a transaction.

        @param tableClass: an Item subclass
        @param key: a 2-tuple of the tableClass's typeName and schemaVersion

        @return: a typeID for the table; a new one if no table exists, or the
        existing one if the table was created by another Store object
        referencing this database.
        """
        try:
            self._justCreateTable(tableClass)
        except errors.TableAlreadyExists:
            # Although we don't have a memory of this table from the last time
            # we called "_startup()", another process has updated the schema
            # since then.
            self._startup()
            return self.typenameAndVersionToID[key]


        typeID = self.executeSchemaSQL(_schema.CREATE_TYPE,
                                       [tableClass.typeName,
                                        tableClass.__module__,
                                        tableClass.schemaVersion])

        self.typenameAndVersionToID[key] = typeID

        if self.tablesCreatedThisTransaction is not None:
            self.tablesCreatedThisTransaction.append(tableClass)

        # If the new type is a legacy type (not the current version), we need
        # to queue it for upgrade to ensure that if we are in the middle of an
        # upgrade, legacy items of this version get upgraded.
        cls = _typeNameToMostRecentClass.get(tableClass.typeName)
        if cls is not None and tableClass.schemaVersion != cls.schemaVersion:
            self._upgradeManager.queueTypeUpgrade(tableClass)

        # We can pass () for extantIndexes here because since the table didn't
        # exist for tableClass, none of its indexes could have either.
        # Whatever checks _createIndexesFor will make would give the same
        # result against the actual set of existing indexes as they will
        # against ().
        self._createIndexesFor(tableClass, ())

        for n, (name, storedAttribute) in enumerate(tableClass.getSchema()):
            self.executeSchemaSQL(
                _schema.ADD_SCHEMA_ATTRIBUTE,
                [typeID, n, storedAttribute.indexed, storedAttribute.sqltype,
                 storedAttribute.allowNone, storedAttribute.attrname,
                 storedAttribute.doc, storedAttribute.__class__.__name__])
            # XXX probably need something better for pythontype eventually,
            # when we figure out a good way to do user-defined attributes or we
            # start parameterizing references.

        return typeID