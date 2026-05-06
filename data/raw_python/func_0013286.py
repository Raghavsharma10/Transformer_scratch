def populateFromRow(self, variantSetRecord):
        """
        Populates this VariantSet from the specified DB row.
        """
        self._created = variantSetRecord.created
        self._updated = variantSetRecord.updated
        self.setAttributesJson(variantSetRecord.attributes)
        self._chromFileMap = {}
        # We can't load directly as we want tuples to be stored
        # rather than lists.
        for key, value in json.loads(variantSetRecord.dataurlindexmap).items():
            self._chromFileMap[key] = tuple(value)
        self._metadata = []
        for jsonDict in json.loads(variantSetRecord.metadata):
            metadata = protocol.fromJson(json.dumps(jsonDict),
                                         protocol.VariantSetMetadata)
            self._metadata.append(metadata)