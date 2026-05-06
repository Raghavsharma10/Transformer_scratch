def list_subgroups_global(self, id):
        """
        List subgroups.

        List the immediate OutcomeGroup children of the outcome group. Paginated.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("GET /api/v1/global/outcome_groups/{id}/subgroups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/global/outcome_groups/{id}/subgroups".format(**path), data=data, params=params, all_pages=True)