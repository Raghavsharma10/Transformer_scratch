def runSearchContinuousSets(self, request):
        """
        Returns a SearchContinuousSetsResponse for the specified
        SearchContinuousSetsRequest object.
        """
        return self.runSearchRequest(
            request, protocol.SearchContinuousSetsRequest,
            protocol.SearchContinuousSetsResponse,
            self.continuousSetsGenerator)