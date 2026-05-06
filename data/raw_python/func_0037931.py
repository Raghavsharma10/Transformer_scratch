def update_folder(self, id, hidden=None, lock_at=None, locked=None, name=None, parent_folder_id=None, position=None, unlock_at=None):
        """
        Update folder.

        Updates a folder
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - name
        """The new name of the folder"""
        if name is not None:
            data["name"] = name

        # OPTIONAL - parent_folder_id
        """The id of the folder to move this folder into. The new folder must be in the same context as the original parent folder."""
        if parent_folder_id is not None:
            data["parent_folder_id"] = parent_folder_id

        # OPTIONAL - lock_at
        """The datetime to lock the folder at"""
        if lock_at is not None:
            data["lock_at"] = lock_at

        # OPTIONAL - unlock_at
        """The datetime to unlock the folder at"""
        if unlock_at is not None:
            data["unlock_at"] = unlock_at

        # OPTIONAL - locked
        """Flag the folder as locked"""
        if locked is not None:
            data["locked"] = locked

        # OPTIONAL - hidden
        """Flag the folder as hidden"""
        if hidden is not None:
            data["hidden"] = hidden

        # OPTIONAL - position
        """Set an explicit sort position for the folder"""
        if position is not None:
            data["position"] = position

        self.logger.debug("PUT /api/v1/folders/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/folders/{id}".format(**path), data=data, params=params, single_item=True)