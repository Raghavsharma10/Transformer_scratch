def runSearchVariantAnnotationSets(self, request):
        """
        Runs the specified SearchVariantAnnotationSetsRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchVariantAnnotationSetsRequest,
            protocol.SearchVariantAnnotationSetsResponse,
            self.variantAnnotationSetsGenerator)