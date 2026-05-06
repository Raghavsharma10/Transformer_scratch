def get_folder_groups(self, id, group_id):
        """
        Get folder.

        Returns the details for a folder
        
        You can get the root folder from a context by using 'root' as the :id.
        For example, you could get the root folder for a course like:
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("GET /api/v1/groups/{group_id}/folders/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/folders/{id}".format(**path), data=data, params=params, single_item=True)