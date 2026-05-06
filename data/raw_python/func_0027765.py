def _loadTypeSchema(self):
        """
        Load all of the stored schema information for all types known by this
        store.  It's important to load everything all at once (rather than
        loading the schema for each type separately as it is needed) to keep
        store opening fast.  A single query with many results is much faster
        than many queries with a few results each.

        @return: A dict with two-tuples of item type name and schema version as
            keys and lists of five-tuples of attribute schema information for
            that type.  The elements of the five-tuple are::

              - a string giving the name of the Python attribute
              - a string giving the SQL type
              - a boolean indicating whether the attribute is indexed
              - the Python attribute type object (eg, axiom.attributes.integer)
              - a string giving documentation for the attribute
        """

        # Oops, need an index going the other way.  This only happens once per
        # store open, and it's based on data queried from the store, so there
        # doesn't seem to be any broader way to cache and re-use the result.
        # However, if we keyed the resulting dict on the database typeID rather
        # than (typeName, schemaVersion), we wouldn't need the information this
        # dict gives us.  That would mean changing the callers of this function
        # to use typeID instead of that tuple, which may be possible.  Probably
        # only represents a very tiny possible speedup.
        typeIDToNameAndVersion = {}
        for key, value in self.typenameAndVersionToID.iteritems():
            typeIDToNameAndVersion[value] = key

        # Indexing attribute, ordering by it, and getting rid of row_offset
        # from the schema and the sorted() here doesn't seem to be any faster
        # than doing this.
        persistedSchema = sorted(self.querySchemaSQL(
            "SELECT attribute, type_id, sqltype, indexed, "
            "pythontype, docstring FROM *DATABASE*.axiom_attributes "))

        # This is trivially (but measurably!) faster than getattr(attributes,
        # pythontype).
        getAttribute = attributes.__dict__.__getitem__

        result = {}
        for (attribute, typeID, sqltype, indexed, pythontype,
             docstring) in persistedSchema:
            key = typeIDToNameAndVersion[typeID]
            if key not in result:
                result[key] = []
            result[key].append((
                    attribute, sqltype, indexed,
                    getAttribute(pythontype), docstring))
        return result