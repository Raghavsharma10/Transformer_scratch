def remove_feature_flag_users(self, user_id, feature):
        """
        Remove feature flag.

        Remove feature flag for a given Account, Course, or User.  (Note that the flag must
        be defined on the Account, Course, or User directly.)  The object will then inherit
        the feature flags from a higher account, if any exist.  If this flag was 'on' or 'off',
        then lower-level account flags that were masked by this one will apply again.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - PATH - feature
        """ID"""
        path["feature"] = feature

        self.logger.debug("DELETE /api/v1/users/{user_id}/features/flags/{feature} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/users/{user_id}/features/flags/{feature}".format(**path), data=data, params=params, single_item=True)