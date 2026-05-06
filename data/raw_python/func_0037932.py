def create_folder_courses(self, name, course_id, hidden=None, lock_at=None, locked=None, parent_folder_id=None, parent_folder_path=None, position=None, unlock_at=None):
        """
        Create folder.

        Creates a folder in the specified context
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - name
        """The name of the folder"""
        data["name"] = name

        # OPTIONAL - parent_folder_id
        """The id of the folder to store the file in. If this and parent_folder_path are sent an error will be returned. If neither is given, a default folder will be used."""
        if parent_folder_id is not None:
            data["parent_folder_id"] = parent_folder_id

        # OPTIONAL - parent_folder_path
        """The path of the folder to store the new folder in. The path separator is the forward slash `/`, never a back slash. The parent folder will be created if it does not already exist. This parameter only applies to new folders in a context that has folders, such as a user, a course, or a group. If this and parent_folder_id are sent an error will be returned. If neither is given, a default folder will be used."""
        if parent_folder_path is not None:
            data["parent_folder_path"] = parent_folder_path

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

        self.logger.debug("POST /api/v1/courses/{course_id}/folders with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/folders".format(**path), data=data, params=params, single_item=True)