def populateFromRow(self, callSetRecord):
        """
        Populates this CallSet from the specified DB row.
        """
        self._biosampleId = callSetRecord.biosampleid
        self.setAttributesJson(callSetRecord.attributes)