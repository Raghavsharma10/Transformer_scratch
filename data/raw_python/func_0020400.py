def to_facets(self, facets, start=0, page_size=None):
        """
        Query the facets results for this query using the specified list of facets with the given start and pageSize

        @param List[Facet] facets: List of facets
        @param int start:  Start index for paging
        @param page_size: Paging PageSize. If set, overrides Facet.max_result
        """

        if len(facets) == 0:
            raise ValueError("Facets must contain at least one entry", "facets")
        str_query = self.__str__()
        facet_query = FacetQuery(str_query, None, facets, start, page_size, query_parameters=self.query_parameters,
                                 wait_for_non_stale_results=self.wait_for_non_stale_results,
                                 wait_for_non_stale_results_timeout=self.timeout, cutoff_etag=self.cutoff_etag)

        command = GetFacetsCommand(query=facet_query)
        return self.session.requests_executor.execute(command)