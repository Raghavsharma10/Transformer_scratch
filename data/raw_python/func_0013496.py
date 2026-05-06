def runSearchVariantSets(self, request):
        """
        Runs the specified SearchVariantSetsRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchVariantSetsRequest,
            protocol.SearchVariantSetsResponse,
            self.variantSetsGenerator)