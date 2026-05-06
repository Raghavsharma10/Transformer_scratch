def list_preferences_type(self, type, user_id, address):
        """
        List preferences.

        Fetch all preferences for the given communication channel
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - PATH - type
        """ID"""
        path["type"] = type

        # REQUIRED - PATH - address
        """ID"""
        path["address"] = address

        self.logger.debug("GET /api/v1/users/{user_id}/communication_channels/{type}/{address}/notification_preferences with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/communication_channels/{type}/{address}/notification_preferences".format(**path), data=data, params=params, all_pages=True)