def list_group_categories_for_context_accounts(self, account_id):
        """
        List group categories for a context.

        Returns a list of group categories in a context
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        self.logger.debug("GET /api/v1/accounts/{account_id}/group_categories with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/group_categories".format(**path), data=data, params=params, all_pages=True)