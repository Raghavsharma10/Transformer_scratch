def remove_usage_rights_groups(self, group_id, file_ids, folder_ids=None):
        """
        Remove usage rights.

        Removes copyright and license information associated with one or more files
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - file_ids
        """List of ids of files to remove associated usage rights from."""
        params["file_ids"] = file_ids

        # OPTIONAL - folder_ids
        """List of ids of folders. Usage rights will be removed from all files in these folders."""
        if folder_ids is not None:
            params["folder_ids"] = folder_ids

        self.logger.debug("DELETE /api/v1/groups/{group_id}/usage_rights with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/groups/{group_id}/usage_rights".format(**path), data=data, params=params, no_data=True)