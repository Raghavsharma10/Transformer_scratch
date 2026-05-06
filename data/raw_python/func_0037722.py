def get_preference_type(self, type, user_id, address, notification):
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

        # REQUIRED - PATH - type
        """ID"""
        path["type"] = type

        # REQUIRED - PATH - address
        """ID"""
        path["address"] = address

        # REQUIRED - PATH - notification
        """ID"""
        path["notification"] = notification

        self.logger.debug("GET /api/v1/users/{user_id}/communication_channels/{type}/{address}/notification_preferences/{notification} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/communication_channels/{type}/{address}/notification_preferences/{notification}".format(**path), data=data, params=params, single_item=True)