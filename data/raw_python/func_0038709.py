def get_sub_accounts_of_account(self, account_id, recursive=None):
        """
        Get the sub-accounts of an account.

        List accounts that are sub-accounts of the given account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - recursive
        """If true, the entire account tree underneath
        this account will be returned (though still paginated). If false, only
        direct sub-accounts of this account will be returned. Defaults to false."""
        if recursive is not None:
            params["recursive"] = recursive

        self.logger.debug("GET /api/v1/accounts/{account_id}/sub_accounts with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/sub_accounts".format(**path), data=data, params=params, all_pages=True)