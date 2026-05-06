def runSearchContinuous(self, request):
        """
        Returns a SearchContinuousResponse for the specified
        SearchContinuousRequest object.

        :param request: JSON string representing searchContinuousRequest
        :return: JSON string representing searchContinuousResponse
        """
        return self.runSearchRequest(
            request, protocol.SearchContinuousRequest,
            protocol.SearchContinuousResponse,
            self.continuousGenerator)