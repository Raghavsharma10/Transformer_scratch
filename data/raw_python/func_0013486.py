def runGetVariantAnnotationSet(self, id_):
        """
        Runs a getVariantSet request for the specified ID.
        """
        compoundId = datamodel.VariantAnnotationSetCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(compoundId.variant_set_id)
        variantAnnotationSet = variantSet.getVariantAnnotationSet(id_)
        return self.runGetRequest(variantAnnotationSet)