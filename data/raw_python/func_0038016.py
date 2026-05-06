def list_groups_available_in_context_accounts(self, account_id, include=None, only_own_groups=None):
        """
        List the groups available in a context.

        Returns the list of active groups in the given context that are visible to user.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - only_own_groups
        """Will only include groups that the user belongs to if this is set"""
        if only_own_groups is not None:
            params["only_own_groups"] = only_own_groups

        # OPTIONAL - include
        """- "tabs": Include the list of tabs configured for each group.  See the
          {api:TabsController#index List available tabs API} for more information."""
        if include is not None:
            self._validate_enum(include, ["tabs"])
            params["include"] = include

        self.logger.debug("GET /api/v1/accounts/{account_id}/groups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/groups".format(**path), data=data, params=params, all_pages=True)