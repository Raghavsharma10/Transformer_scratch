def get_single_group_category(self, group_category_id):
        """
        Get a single group category.

        Returns the data for a single group category, or a 401 if the caller doesn't have
        the rights to see it.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_category_id
        """ID"""
        path["group_category_id"] = group_category_id

        self.logger.debug("GET /api/v1/group_categories/{group_category_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/group_categories/{group_category_id}".format(**path), data=data, params=params, single_item=True)