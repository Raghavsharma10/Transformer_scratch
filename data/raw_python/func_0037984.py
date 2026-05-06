def list_enabled_features_accounts(self, account_id):
        """
        List enabled features.

        List all features that are enabled on a given Account, Course, or User.
        Only the feature names are returned.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        self.logger.debug("GET /api/v1/accounts/{account_id}/features/enabled with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/features/enabled".format(**path), data=data, params=params, no_data=True)