def runGetVariantSet(self, id_):
        """
        Runs a getVariantSet request for the specified ID.
        """
        compoundId = datamodel.VariantSetCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(id_)
        return self.runGetRequest(variantSet)