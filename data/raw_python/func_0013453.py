def variantAnnotationSetsGenerator(self, request):
        """
        Returns a generator over the (variantAnnotationSet, nextPageToken)
        pairs defined by the specified request.
        """
        compoundId = datamodel.VariantSetCompoundId.parse(
            request.variant_set_id)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(request.variant_set_id)
        return self._topLevelObjectGenerator(
            request, variantSet.getNumVariantAnnotationSets(),
            variantSet.getVariantAnnotationSetByIndex)