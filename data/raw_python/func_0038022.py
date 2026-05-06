def list_group_memberships(self, group_id, filter_states=None):
        """
        List group memberships.

        List the members of a group.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # OPTIONAL - filter_states
        """Only list memberships with the given workflow_states. By default it will
        return all memberships."""
        if filter_states is not None:
            self._validate_enum(filter_states, ["accepted", "invited", "requested"])
            params["filter_states"] = filter_states

        self.logger.debug("GET /api/v1/groups/{group_id}/memberships with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/memberships".format(**path), data=data, params=params, all_pages=True)