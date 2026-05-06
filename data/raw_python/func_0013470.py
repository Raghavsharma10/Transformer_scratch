def runGetCallSet(self, id_):
        """
        Returns a callset with the given id
        """
        compoundId = datamodel.CallSetCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(compoundId.variant_set_id)
        callSet = variantSet.getCallSet(id_)
        return self.runGetRequest(callSet)