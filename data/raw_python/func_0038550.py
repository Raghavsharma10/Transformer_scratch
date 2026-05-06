def list_conferences_groups(self, group_id):
        """
        List conferences.

        Retrieve the list of conferences for this context
        
        This API returns a JSON object containing the list of conferences,
        the key for the list of conferences is "conferences"
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        self.logger.debug("GET /api/v1/groups/{group_id}/conferences with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/conferences".format(**path), data=data, params=params, all_pages=True)