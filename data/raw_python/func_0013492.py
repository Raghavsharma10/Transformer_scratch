def runSearchBiosamples(self, request):
        """
        Runs the specified SearchBiosamplesRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchBiosamplesRequest,
            protocol.SearchBiosamplesResponse,
            self.biosamplesGenerator)