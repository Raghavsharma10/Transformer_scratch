def _checkTypeSchemaConsistency(self, actualType, onDiskSchema):
        """
        Called for all known types at database startup: make sure that what we
        know (in memory) about this type agrees with what is stored about this
        type in the database.

        @param actualType: A L{MetaItem} instance which is associated with a
            table in this store.  The schema it defines in memory will be
            checked against the schema known in the database to ensure they
            agree.

        @param onDiskSchema: A mapping from L{MetaItem} instances (such as
            C{actualType}) to the schema known in the database and associated
            with C{actualType}.

        @raise RuntimeError: if the schema defined by C{actualType} does not
            match the database-present schema given in C{onDiskSchema} or if
            C{onDiskSchema} contains a newer version of the schema associated
            with C{actualType} than C{actualType} represents.
        """
        # make sure that both the runtime and the database both know about this
        # type; if they don't both know, we can't check that their views are
        # consistent
        try:
            inMemorySchema = _inMemorySchemaCache[actualType]
        except KeyError:
            inMemorySchema = _inMemorySchemaCache[actualType] = [
                (storedAttribute.attrname, storedAttribute.sqltype)
                for (name, storedAttribute) in actualType.getSchema()]

        key = (actualType.typeName, actualType.schemaVersion)
        persistedSchema = [(storedAttribute[0], storedAttribute[1])
                           for storedAttribute in onDiskSchema[key]]
        if inMemorySchema != persistedSchema:
            raise RuntimeError(
                "Schema mismatch on already-loaded %r <%r> object version %d:\n%s" %
                (actualType, actualType.typeName, actualType.schemaVersion,
                 _diffSchema(persistedSchema, inMemorySchema)))

        if actualType.__legacy__:
            return

        if (key[0], key[1] + 1) in onDiskSchema:
            raise RuntimeError(
                "Memory version of %r is %d; database has newer" % (
                    actualType.typeName, key[1]))