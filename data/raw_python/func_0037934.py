def copy_file(self, dest_folder_id, source_file_id, on_duplicate=None):
        """
        Copy a file.

        Copy a file from elsewhere in Canvas into a folder.
        
        Copying a file across contexts (between courses and users) is permitted,
        but the source and destination must belong to the same institution.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - dest_folder_id
        """ID"""
        path["dest_folder_id"] = dest_folder_id

        # REQUIRED - source_file_id
        """The id of the source file"""
        data["source_file_id"] = source_file_id

        # OPTIONAL - on_duplicate
        """What to do if a file with the same name already exists at the destination.
        If such a file exists and this parameter is not given, the call will fail.
        
        "overwrite":: Replace an existing file with the same name
        "rename":: Add a qualifier to make the new filename unique"""
        if on_duplicate is not None:
            self._validate_enum(on_duplicate, ["overwrite", "rename"])
            data["on_duplicate"] = on_duplicate

        self.logger.debug("POST /api/v1/folders/{dest_folder_id}/copy_file with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/folders/{dest_folder_id}/copy_file".format(**path), data=data, params=params, single_item=True)