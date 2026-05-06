def get_single_role(self, id, role_id, account_id, role=None):
        """
        Get a single role.

        Retrieve information about a single role
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - PATH - account_id
        """The id of the account containing the role"""
        path["account_id"] = account_id

        # REQUIRED - role_id
        """The unique identifier for the role"""
        params["role_id"] = role_id

        # OPTIONAL - role
        """The name for the role"""
        if role is not None:
            params["role"] = role

        self.logger.debug("GET /api/v1/accounts/{account_id}/roles/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/roles/{id}".format(**path), data=data, params=params, single_item=True)