def list_preferences_communication_channel_id(self, user_id, communication_channel_id):
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

        # REQUIRED - PATH - communication_channel_id
        """ID"""
        path["communication_channel_id"] = communication_channel_id

        self.logger.debug("GET /api/v1/users/{user_id}/communication_channels/{communication_channel_id}/notification_preferences with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/communication_channels/{communication_channel_id}/notification_preferences".format(**path), data=data, params=params, all_pages=True)