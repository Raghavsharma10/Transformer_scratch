def callSetsGenerator(self, request):
        """
        Returns a generator over the (callSet, nextPageToken) pairs defined
        by the specified request.
        """
        compoundId = datamodel.VariantSetCompoundId.parse(
            request.variant_set_id)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        variantSet = dataset.getVariantSet(compoundId.variant_set_id)
        results = []
        for obj in variantSet.getCallSets():
            include = True
            if request.name:
                if request.name != obj.getLocalId():
                    include = False
            if request.biosample_id:
                if request.biosample_id != obj.getBiosampleId():
                    include = False
            if include:
                results.append(obj)
        return self._objectListGenerator(request, results)