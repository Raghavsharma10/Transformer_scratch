def leave_group_memberships(self, group_id, membership_id):
        """
        Leave a group.

        Leave a group if you are allowed to leave (some groups, such as sets of
        course groups created by teachers, cannot be left). You may also use 'self'
        in place of a membership_id.
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

        self.logger.debug("DELETE /api/v1/groups/{group_id}/memberships/{membership_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/groups/{group_id}/memberships/{membership_id}".format(**path), data=data, params=params, no_data=True)