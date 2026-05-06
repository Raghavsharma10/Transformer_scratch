def variantAnnotationsGenerator(self, request):
        """
        Returns a generator over the (variantAnnotaitons, nextPageToken) pairs
        defined by the specified request.
        """
        compoundId = datamodel.VariantAnnotationSetCompoundId.parse(
            request.variant_annotation_set_id)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(compoundId.variant_set_id)
        variantAnnotationSet = variantSet.getVariantAnnotationSet(
            request.variant_annotation_set_id)
        iterator = paging.VariantAnnotationsIntervalIterator(
            request, variantAnnotationSet)
        return iterator