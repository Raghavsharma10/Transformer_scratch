def continuousGenerator(self, request):
        """
        Returns a generator over the (continuous, nextPageToken) pairs
        defined by the (JSON string) request.
        """
        compoundId = None
        if request.continuous_set_id != "":
            compoundId = datamodel.ContinuousSetCompoundId.parse(
                request.continuous_set_id)
        if compoundId is None:
            raise exceptions.ContinuousSetNotSpecifiedException()

        dataset = self.getDataRepository().getDataset(
            compoundId.dataset_id)
        continuousSet = dataset.getContinuousSet(request.continuous_set_id)
        iterator = paging.ContinuousIterator(request, continuousSet)
        return iterator