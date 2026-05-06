def list_roles(self, account_id, show_inherited=None, state=None):
        """
        List roles.

        List the roles available to an account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """The id of the account to retrieve roles for."""
        path["account_id"] = account_id

        # OPTIONAL - state
        """Filter by role state. If this argument is omitted, only 'active' roles are
        returned."""
        if state is not None:
            self._validate_enum(state, ["active", "inactive"])
            params["state"] = state

        # OPTIONAL - show_inherited
        """If this argument is true, all roles inherited from parent accounts will
        be included."""
        if show_inherited is not None:
            params["show_inherited"] = show_inherited

        self.logger.debug("GET /api/v1/accounts/{account_id}/roles with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/roles".format(**path), data=data, params=params, all_pages=True)