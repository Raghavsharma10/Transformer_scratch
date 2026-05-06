def update_preference_type(self, type, address, notification, notification_preferences_frequency):
        """
        Update a preference.

        Change the preference for a single notification for a single communication channel
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - type
        """ID"""
        path["type"] = type

        # REQUIRED - PATH - address
        """ID"""
        path["address"] = address

        # REQUIRED - PATH - notification
        """ID"""
        path["notification"] = notification

        # REQUIRED - notification_preferences[frequency]
        """The desired frequency for this notification"""
        data["notification_preferences[frequency]"] = notification_preferences_frequency

        self.logger.debug("PUT /api/v1/users/self/communication_channels/{type}/{address}/notification_preferences/{notification} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/users/self/communication_channels/{type}/{address}/notification_preferences/{notification}".format(**path), data=data, params=params, no_data=True)