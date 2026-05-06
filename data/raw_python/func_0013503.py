def runSearchFeatures(self, request):
        """
        Returns a SearchFeaturesResponse for the specified
        SearchFeaturesRequest object.

        :param request: JSON string representing searchFeaturesRequest
        :return: JSON string representing searchFeatureResponse
        """
        return self.runSearchRequest(
            request, protocol.SearchFeaturesRequest,
            protocol.SearchFeaturesResponse,
            self.featuresGenerator)