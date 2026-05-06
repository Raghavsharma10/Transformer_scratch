def runGetReference(self, id_):
        """
        Runs a getReference request for the specified ID.
        """
        compoundId = datamodel.ReferenceCompoundId.parse(id_)
        referenceSet = self.getDataRepository().getReferenceSet(
            compoundId.reference_set_id)
        reference = referenceSet.getReference(id_)
        return self.runGetRequest(reference)