def query_by_account(self, account_id, end_time=None, start_time=None):
        """
        Query by account.

        List authentication events for a given account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - start_time
        """The beginning of the time range from which you want events."""
        if start_time is not None:
            params["start_time"] = start_time

        # OPTIONAL - end_time
        """The end of the time range from which you want events."""
        if end_time is not None:
            params["end_time"] = end_time

        self.logger.debug("GET /api/v1/audit/authentication/accounts/{account_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/audit/authentication/accounts/{account_id}".format(**path), data=data, params=params, no_data=True)