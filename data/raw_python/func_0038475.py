def get_all_outcome_links_for_context_accounts(self, account_id, outcome_group_style=None, outcome_style=None):
        """
        Get all outcome links for context.

        
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - outcome_style
        """The detail level of the outcomes. Defaults to "abbrev".
        Specify "full" for more information."""
        if outcome_style is not None:
            params["outcome_style"] = outcome_style

        # OPTIONAL - outcome_group_style
        """The detail level of the outcome groups. Defaults to "abbrev".
        Specify "full" for more information."""
        if outcome_group_style is not None:
            params["outcome_group_style"] = outcome_group_style

        self.logger.debug("GET /api/v1/accounts/{account_id}/outcome_group_links with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/outcome_group_links".format(**path), data=data, params=params, all_pages=True)