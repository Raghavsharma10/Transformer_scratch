def list_subgroups_accounts(self, id, account_id):
        """
        List subgroups.

        List the immediate OutcomeGroup children of the outcome group. Paginated.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("GET /api/v1/accounts/{account_id}/outcome_groups/{id}/subgroups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/outcome_groups/{id}/subgroups".format(**path), data=data, params=params, all_pages=True)