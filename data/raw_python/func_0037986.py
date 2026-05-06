def set_feature_flag_accounts(self, feature, account_id, state=None):
        """
        Set feature flag.

        Set a feature flag for a given Account, Course, or User. This call will fail if a parent account sets
        a feature flag for the same feature in any state other than "allowed".
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - PATH - feature
        """ID"""
        path["feature"] = feature

        # OPTIONAL - state
        """"off":: The feature is not available for the course, user, or account and sub-accounts.
        "allowed":: (valid only on accounts) The feature is off in the account, but may be enabled in
                    sub-accounts and courses by setting a feature flag on the sub-account or course.
        "on":: The feature is turned on unconditionally for the user, course, or account and sub-accounts."""
        if state is not None:
            self._validate_enum(state, ["off", "allowed", "on"])
            data["state"] = state

        self.logger.debug("PUT /api/v1/accounts/{account_id}/features/flags/{feature} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/accounts/{account_id}/features/flags/{feature}".format(**path), data=data, params=params, single_item=True)