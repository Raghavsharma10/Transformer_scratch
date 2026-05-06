def get_single_group_membership_memberships(self, group_id, membership_id):
        """
        Get a single group membership.

        Returns the group membership with the given membership id or user id.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - PATH - membership_id
        """ID"""
        path["membership_id"] = membership_id

        self.logger.debug("GET /api/v1/groups/{group_id}/memberships/{membership_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/memberships/{membership_id}".format(**path), data=data, params=params, single_item=True)