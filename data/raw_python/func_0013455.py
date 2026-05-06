def variantsGenerator(self, request):
        """
        Returns a generator over the (variant, nextPageToken) pairs defined
        by the specified request.
        """
        compoundId = datamodel.VariantSetCompoundId \
            .parse(request.variant_set_id)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(compoundId.variant_set_id)
        intervalIterator = paging.VariantsIntervalIterator(
            request, variantSet)
        return intervalIterator