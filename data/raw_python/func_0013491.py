def runSearchIndividuals(self, request):
        """
        Runs the specified search SearchIndividualsRequest.
        """
        return self.runSearchRequest(
            request, protocol.SearchIndividualsRequest,
            protocol.SearchIndividualsResponse,
            self.individualsGenerator)