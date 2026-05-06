def show_page_groups(self, url, group_id):
        """
        Show page.

        Retrieve the content of a wiki page
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - PATH - url
        """ID"""
        path["url"] = url

        self.logger.debug("GET /api/v1/groups/{group_id}/pages/{url} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/pages/{url}".format(**path), data=data, params=params, single_item=True)