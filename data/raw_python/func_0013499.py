def runSearchVariantAnnotations(self, request):
        """
        Runs the specified SearchVariantAnnotationsRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchVariantAnnotationsRequest,
            protocol.SearchVariantAnnotationsResponse,
            self.variantAnnotationsGenerator)