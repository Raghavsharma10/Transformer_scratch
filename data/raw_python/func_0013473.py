def runListPeers(self, request):
        """
        Takes a ListPeersRequest and returns a ListPeersResponse using
        a page_token and page_size if provided.
        """
        return self.runSearchRequest(
            request,
            protocol.ListPeersRequest,
            protocol.ListPeersResponse,
            self.peersGenerator)