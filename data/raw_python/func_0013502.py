def runSearchFeatureSets(self, request):
        """
        Returns a SearchFeatureSetsResponse for the specified
        SearchFeatureSetsRequest object.
        """
        return self.runSearchRequest(
            request, protocol.SearchFeatureSetsRequest,
            protocol.SearchFeatureSetsResponse,
            self.featureSetsGenerator)