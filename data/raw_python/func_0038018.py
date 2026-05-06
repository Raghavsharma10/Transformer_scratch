def edit_group(self, group_id, avatar_id=None, description=None, is_public=None, join_level=None, members=None, name=None, storage_quota_mb=None):
        """
        Edit a group.

        Modifies an existing group.  Note that to set an avatar image for the
        group, you must first upload the image file to the group, and the use the
        id in the response as the argument to this function.  See the
        {file:file_uploads.html File Upload Documentation} for details on the file
        upload workflow.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # OPTIONAL - name
        """The name of the group"""
        if name is not None:
            data["name"] = name

        # OPTIONAL - description
        """A description of the group"""
        if description is not None:
            data["description"] = description

        # OPTIONAL - is_public
        """Whether the group is public (applies only to community groups). Currently
        you cannot set a group back to private once it has been made public."""
        if is_public is not None:
            data["is_public"] = is_public

        # OPTIONAL - join_level
        """no description"""
        if join_level is not None:
            self._validate_enum(join_level, ["parent_context_auto_join", "parent_context_request", "invitation_only"])
            data["join_level"] = join_level

        # OPTIONAL - avatar_id
        """The id of the attachment previously uploaded to the group that you would
        like to use as the avatar image for this group."""
        if avatar_id is not None:
            data["avatar_id"] = avatar_id

        # OPTIONAL - storage_quota_mb
        """The allowed file storage for the group, in megabytes. This parameter is
        ignored if the caller does not have the manage_storage_quotas permission."""
        if storage_quota_mb is not None:
            data["storage_quota_mb"] = storage_quota_mb

        # OPTIONAL - members
        """An array of user ids for users you would like in the group.
        Users not in the group will be sent invitations. Existing group
        members who aren't in the list will be removed from the group."""
        if members is not None:
            data["members"] = members

        self.logger.debug("PUT /api/v1/groups/{group_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/groups/{group_id}".format(**path), data=data, params=params, single_item=True)