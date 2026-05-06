def runSearchReads(self, request):
        """
        Runs the specified SearchReadsRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchReadsRequest,
            protocol.SearchReadsResponse,
            self.readsGenerator)