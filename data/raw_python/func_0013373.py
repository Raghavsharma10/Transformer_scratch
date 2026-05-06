def insertVariantSet(self, variantSet):
        """
        Inserts a the specified variantSet into this repository.
        """
        # We cheat a little here with the VariantSetMetadata, and encode these
        # within the table as a JSON dump. These should really be stored in
        # their own table
        metadataJson = json.dumps(
            [protocol.toJsonDict(metadata) for metadata in
             variantSet.getMetadata()])
        urlMapJson = json.dumps(variantSet.getReferenceToDataUrlIndexMap())
        try:
            models.Variantset.create(
                id=variantSet.getId(),
                datasetid=variantSet.getParentContainer().getId(),
                referencesetid=variantSet.getReferenceSet().getId(),
                name=variantSet.getLocalId(),
                created=datetime.datetime.now(),
                updated=datetime.datetime.now(),
                metadata=metadataJson,
                dataurlindexmap=urlMapJson,
                attributes=json.dumps(variantSet.getAttributes()))
        except Exception as e:
            raise exceptions.RepoManagerException(e)
        for callSet in variantSet.getCallSets():
            self.insertCallSet(callSet)