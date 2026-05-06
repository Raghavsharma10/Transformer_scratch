def list_pages_groups(self, group_id, order=None, published=None, search_term=None, sort=None):
        """
        List pages.

        List the wiki pages associated with a course or group
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # OPTIONAL - sort
        """Sort results by this field."""
        if sort is not None:
            self._validate_enum(sort, ["title", "created_at", "updated_at"])
            params["sort"] = sort

        # OPTIONAL - order
        """The sorting order. Defaults to 'asc'."""
        if order is not None:
            self._validate_enum(order, ["asc", "desc"])
            params["order"] = order

        # OPTIONAL - search_term
        """The partial title of the pages to match and return."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - published
        """If true, include only published paqes. If false, exclude published
        pages. If not present, do not filter on published status."""
        if published is not None:
            params["published"] = published

        self.logger.debug("GET /api/v1/groups/{group_id}/pages with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/pages".format(**path), data=data, params=params, all_pages=True)