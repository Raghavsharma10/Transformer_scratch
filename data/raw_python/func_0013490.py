def runSearchReadGroupSets(self, request):
        """
        Runs the specified SearchReadGroupSetsRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchReadGroupSetsRequest,
            protocol.SearchReadGroupSetsResponse,
            self.readGroupSetsGenerator)