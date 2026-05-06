def list_revisions_groups(self, url, group_id):
        """
        List revisions.

        List the revisions of a page. Callers must have update rights on the page in order to see page history.
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

        self.logger.debug("GET /api/v1/groups/{group_id}/pages/{url}/revisions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/pages/{url}/revisions".format(**path), data=data, params=params, all_pages=True)