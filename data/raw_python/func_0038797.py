def redirect_to_assignment_override_for_group(self, group_id, assignment_id):
        """
        Redirect to the assignment override for a group.

        Responds with a redirect to the override for the given group, if any
        (404 otherwise).
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        self.logger.debug("GET /api/v1/groups/{group_id}/assignments/{assignment_id}/override with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/assignments/{assignment_id}/override".format(**path), data=data, params=params, no_data=True)