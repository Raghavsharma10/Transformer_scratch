def list_account_admins(self, account_id, user_id=None):
        """
        List account admins.

        List the admins in the account
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - user_id
        """Scope the results to those with user IDs equal to any of the IDs specified here."""
        if user_id is not None:
            params["user_id"] = user_id

        self.logger.debug("GET /api/v1/accounts/{account_id}/admins with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/admins".format(**path), data=data, params=params, all_pages=True)