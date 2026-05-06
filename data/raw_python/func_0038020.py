def list_group_s_users(self, group_id, include=None, search_term=None):
        """
        List group's users.

        Returns a list of users in the group.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # OPTIONAL - search_term
        """The partial name or full ID of the users to match and return in the
        results list. Must be at least 3 characters."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - include
        """- "avatar_url": Include users' avatar_urls."""
        if include is not None:
            self._validate_enum(include, ["avatar_url"])
            params["include"] = include

        self.logger.debug("GET /api/v1/groups/{group_id}/users with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/users".format(**path), data=data, params=params, all_pages=True)