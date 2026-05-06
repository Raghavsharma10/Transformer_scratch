def list_users_in_group_category(self, group_category_id, search_term=None, unassigned=None):
        """
        List users in group category.

        Returns a list of users in the group category.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_category_id
        """ID"""
        path["group_category_id"] = group_category_id

        # OPTIONAL - search_term
        """The partial name or full ID of the users to match and return in the results
        list. Must be at least 3 characters."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - unassigned
        """Set this value to true if you wish only to search unassigned users in the
        group category."""
        if unassigned is not None:
            params["unassigned"] = unassigned

        self.logger.debug("GET /api/v1/group_categories/{group_category_id}/users with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/group_categories/{group_category_id}/users".format(**path), data=data, params=params, all_pages=True)