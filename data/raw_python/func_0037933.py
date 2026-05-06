def delete_folder(self, id, force=None):
        """
        Delete folder.

        Remove the specified folder. You can only delete empty folders unless you
        set the 'force' flag
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - force
        """Set to 'true' to allow deleting a non-empty folder"""
        if force is not None:
            params["force"] = force

        self.logger.debug("DELETE /api/v1/folders/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/folders/{id}".format(**path), data=data, params=params, no_data=True)