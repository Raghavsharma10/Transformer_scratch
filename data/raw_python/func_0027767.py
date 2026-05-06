def _prepareOldVersionOf(self, typename, version, persistedSchema):
        """
        Note that this database contains old versions of a particular type.
        Create the appropriate dummy item subclass and queue the type to be
        upgraded.

        @param typename: The I{typeName} associated with the schema for which
            to create a dummy item class.

        @param version: The I{schemaVersion} of the old version of the schema
            for which to create a dummy item class.

        @param persistedSchema: A mapping giving information about all schemas
            stored in the database, used to create the attributes of the dummy
            item class.
        """
        appropriateSchema = persistedSchema[typename, version]
        # create actual attribute objects
        dummyAttributes = {}
        for (attribute, sqlType, indexed, pythontype,
             docstring) in appropriateSchema:
            atr = pythontype(indexed=indexed, doc=docstring)
            dummyAttributes[attribute] = atr
        dummyBases = []
        oldType = declareLegacyItem(
            typename, version, dummyAttributes, dummyBases)
        self._upgradeManager.queueTypeUpgrade(oldType)
        return oldType