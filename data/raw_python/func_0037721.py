def get_preference_communication_channel_id(self, user_id, notification, communication_channel_id):
        """
        Get a preference.

        Fetch the preference for the given notification for the given communicaiton channel
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

        # REQUIRED - PATH - notification
        """ID"""
        path["notification"] = notification

        self.logger.debug("GET /api/v1/users/{user_id}/communication_channels/{communication_channel_id}/notification_preferences/{notification} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/communication_channels/{communication_channel_id}/notification_preferences/{notification}".format(**path), data=data, params=params, single_item=True)