def runGetVariant(self, id_):
        """
        Returns a variant with the given id
        """
        compoundId = datamodel.VariantCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(compoundId.variant_set_id)
        gaVariant = variantSet.getVariant(compoundId)
        # TODO variant is a special case here, as it's returning a
        # protocol element rather than a datamodel object. We should
        # fix this for consistency.
        jsonString = protocol.toJson(gaVariant)
        return jsonString