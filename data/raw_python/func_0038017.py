def create_group_groups(self, description=None, is_public=None, join_level=None, name=None, storage_quota_mb=None):
        """
        Create a group.

        Creates a new group. Groups created using the "/api/v1/groups/"
        endpoint will be community groups.
        """
        path = {}
        data = {}
        params = {}

        # OPTIONAL - name
        """The name of the group"""
        if name is not None:
            data["name"] = name

        # OPTIONAL - description
        """A description of the group"""
        if description is not None:
            data["description"] = description

        # OPTIONAL - is_public
        """whether the group is public (applies only to community groups)"""
        if is_public is not None:
            data["is_public"] = is_public

        # OPTIONAL - join_level
        """no description"""
        if join_level is not None:
            self._validate_enum(join_level, ["parent_context_auto_join", "parent_context_request", "invitation_only"])
            data["join_level"] = join_level

        # OPTIONAL - storage_quota_mb
        """The allowed file storage for the group, in megabytes. This parameter is
        ignored if the caller does not have the manage_storage_quotas permission."""
        if storage_quota_mb is not None:
            data["storage_quota_mb"] = storage_quota_mb

        self.logger.debug("POST /api/v1/groups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/groups".format(**path), data=data, params=params, single_item=True)