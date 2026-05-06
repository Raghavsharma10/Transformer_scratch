def create_group_category_accounts(self, name, account_id, auto_leader=None, create_group_count=None, group_limit=None, self_signup=None, split_group_count=None):
        """
        Create a Group Category.

        Create a new group category
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - name
        """Name of the group category"""
        data["name"] = name

        # OPTIONAL - self_signup
        """Allow students to sign up for a group themselves (Course Only).
        valid values are:
        "enabled":: allows students to self sign up for any group in course
        "restricted":: allows students to self sign up only for groups in the
                       same section null disallows self sign up"""
        if self_signup is not None:
            self._validate_enum(self_signup, ["enabled", "restricted"])
            data["self_signup"] = self_signup

        # OPTIONAL - auto_leader
        """Assigns group leaders automatically when generating and allocating students to groups
        Valid values are:
        "first":: the first student to be allocated to a group is the leader
        "random":: a random student from all members is chosen as the leader"""
        if auto_leader is not None:
            self._validate_enum(auto_leader, ["first", "random"])
            data["auto_leader"] = auto_leader

        # OPTIONAL - group_limit
        """Limit the maximum number of users in each group (Course Only). Requires
        self signup."""
        if group_limit is not None:
            data["group_limit"] = group_limit

        # OPTIONAL - create_group_count
        """Create this number of groups (Course Only)."""
        if create_group_count is not None:
            data["create_group_count"] = create_group_count

        # OPTIONAL - split_group_count
        """(Deprecated)
        Create this number of groups, and evenly distribute students
        among them. not allowed with "enable_self_signup". because
        the group assignment happens synchronously, it's recommended
        that you instead use the assign_unassigned_members endpoint.
        (Course Only)"""
        if split_group_count is not None:
            data["split_group_count"] = split_group_count

        self.logger.debug("POST /api/v1/accounts/{account_id}/group_categories with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/group_categories".format(**path), data=data, params=params, single_item=True)