def featuresGenerator(self, request):
        """
        Returns a generator over the (features, nextPageToken) pairs
        defined by the (JSON string) request.
        """
        compoundId = None
        parentId = None
        if request.feature_set_id != "":
            compoundId = datamodel.FeatureSetCompoundId.parse(
                request.feature_set_id)
        if request.parent_id != "":
            compoundParentId = datamodel.FeatureCompoundId.parse(
                request.parent_id)
            parentId = compoundParentId.featureId
            # A client can optionally specify JUST the (compound) parentID,
            # and the server needs to derive the dataset & featureSet
            # from this (compound) parentID.
            if compoundId is None:
                compoundId = compoundParentId
            else:
                # check that the dataset and featureSet of the parent
                # compound ID is the same as that of the featureSetId
                mismatchCheck = (
                    compoundParentId.dataset_id != compoundId.dataset_id or
                    compoundParentId.feature_set_id !=
                    compoundId.feature_set_id)
                if mismatchCheck:
                    raise exceptions.ParentIncompatibleWithFeatureSet()

        if compoundId is None:
            raise exceptions.FeatureSetNotSpecifiedException()

        dataset = self.getDataRepository().getDataset(
            compoundId.dataset_id)
        featureSet = dataset.getFeatureSet(compoundId.feature_set_id)
        iterator = paging.FeaturesIterator(
            request, featureSet, parentId)
        return iterator