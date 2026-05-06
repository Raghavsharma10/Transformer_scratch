def show_front_page_groups(self, group_id):
        """
        Show front page.

        Retrieve the content of the front page
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        self.logger.debug("GET /api/v1/groups/{group_id}/front_page with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/front_page".format(**path), data=data, params=params, single_item=True)