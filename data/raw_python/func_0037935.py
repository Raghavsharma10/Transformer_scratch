def copy_folder(self, dest_folder_id, source_folder_id):
        """
        Copy a folder.

        Copy a folder (and its contents) from elsewhere in Canvas into a folder.
        
        Copying a folder across contexts (between courses and users) is permitted,
        but the source and destination must belong to the same institution.
        If the source and destination folders are in the same context, the
        source folder may not contain the destination folder. A folder will be
        renamed at its destination if another folder with the same name already
        exists.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - dest_folder_id
        """ID"""
        path["dest_folder_id"] = dest_folder_id

        # REQUIRED - source_folder_id
        """The id of the source folder"""
        data["source_folder_id"] = source_folder_id

        self.logger.debug("POST /api/v1/folders/{dest_folder_id}/copy_folder with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/folders/{dest_folder_id}/copy_folder".format(**path), data=data, params=params, single_item=True)