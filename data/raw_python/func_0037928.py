def update_file(self, id, hidden=None, lock_at=None, locked=None, name=None, on_duplicate=None, parent_folder_id=None, unlock_at=None):
        """
        Update file.

        Update some settings on the specified file
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - name
        """The new display name of the file"""
        if name is not None:
            data["name"] = name

        # OPTIONAL - parent_folder_id
        """The id of the folder to move this file into.
        The new folder must be in the same context as the original parent folder.
        If the file is in a context without folders this does not apply."""
        if parent_folder_id is not None:
            data["parent_folder_id"] = parent_folder_id

        # OPTIONAL - on_duplicate
        """If the file is moved to a folder containing a file with the same name,
        or renamed to a name matching an existing file, the API call will fail
        unless this parameter is supplied.
        
        "overwrite":: Replace the existing file with the same name
        "rename":: Add a qualifier to make the new filename unique"""
        if on_duplicate is not None:
            self._validate_enum(on_duplicate, ["overwrite", "rename"])
            data["on_duplicate"] = on_duplicate

        # OPTIONAL - lock_at
        """The datetime to lock the file at"""
        if lock_at is not None:
            data["lock_at"] = lock_at

        # OPTIONAL - unlock_at
        """The datetime to unlock the file at"""
        if unlock_at is not None:
            data["unlock_at"] = unlock_at

        # OPTIONAL - locked
        """Flag the file as locked"""
        if locked is not None:
            data["locked"] = locked

        # OPTIONAL - hidden
        """Flag the file as hidden"""
        if hidden is not None:
            data["hidden"] = hidden

        self.logger.debug("PUT /api/v1/files/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/files/{id}".format(**path), data=data, params=params, single_item=True)