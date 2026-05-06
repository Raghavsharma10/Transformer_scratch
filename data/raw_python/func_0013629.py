def _formatFilterQuery(self, request=None, featureSets=[]):
        """
        Generate a formatted sparql query with appropriate filters
        """
        query = self._baseQuery()
        filters = []
        if issubclass(request.__class__,
                      protocol.SearchGenotypePhenotypeRequest):
            filters += self._filterSearchGenotypePhenotypeRequest(
                request, featureSets)

        if issubclass(request.__class__, protocol.SearchPhenotypesRequest):
            filters += self._filterSearchPhenotypesRequest(request)

        # apply filters
        filter = "FILTER ({})".format(' && '.join(filters))
        if len(filters) == 0:
            filter = ""
        query = query.replace("#%FILTER%", filter)
        return query