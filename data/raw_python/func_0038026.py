def leave_group_users(self, user_id, group_id):
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

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        self.logger.debug("DELETE /api/v1/groups/{group_id}/users/{user_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/groups/{group_id}/users/{user_id}".format(**path), data=data, params=params, no_data=True)