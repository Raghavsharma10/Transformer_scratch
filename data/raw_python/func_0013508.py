def runSearchExpressionLevels(self, request):
        """
        Returns a SearchExpressionLevelResponse for the specified
        SearchExpressionLevelRequest object.
        """
        return self.runSearchRequest(
            request, protocol.SearchExpressionLevelsRequest,
            protocol.SearchExpressionLevelsResponse,
            self.expressionLevelsGenerator)