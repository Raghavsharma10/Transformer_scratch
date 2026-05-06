def list_groups_in_group_category(self, group_category_id):
        """
        List groups in group category.

        Returns a list of groups in a group category
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_category_id
        """ID"""
        path["group_category_id"] = group_category_id

        self.logger.debug("GET /api/v1/group_categories/{group_category_id}/groups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/group_categories/{group_category_id}/groups".format(**path), data=data, params=params, all_pages=True)