def runSearchDatasets(self, request):
        """
        Runs the specified SearchDatasetsRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchDatasetsRequest,
            protocol.SearchDatasetsResponse,
            self.datasetsGenerator)