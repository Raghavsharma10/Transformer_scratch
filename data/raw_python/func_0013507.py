def runSearchRnaQuantifications(self, request):
        """
        Returns a SearchRnaQuantificationResponse for the specified
        SearchRnaQuantificationRequest object.
        """
        return self.runSearchRequest(
            request, protocol.SearchRnaQuantificationsRequest,
            protocol.SearchRnaQuantificationsResponse,
            self.rnaQuantificationsGenerator)