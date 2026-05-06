def runSearchRnaQuantificationSets(self, request):
        """
        Returns a SearchRnaQuantificationSetsResponse for the specified
        SearchRnaQuantificationSetsRequest object.
        """
        return self.runSearchRequest(
            request, protocol.SearchRnaQuantificationSetsRequest,
            protocol.SearchRnaQuantificationSetsResponse,
            self.rnaQuantificationSetsGenerator)