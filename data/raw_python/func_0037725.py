def update_preferences_by_category(self, category, communication_channel_id, notification_preferences_frequency):
        """
        Update preferences by category.

        Change the preferences for multiple notifications based on the category for a single communication channel
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - communication_channel_id
        """ID"""
        path["communication_channel_id"] = communication_channel_id

        # REQUIRED - PATH - category
        """The name of the category. Must be parameterized (e.g. The category "Course Content" should be "course_content")"""
        path["category"] = category

        # REQUIRED - notification_preferences[frequency]
        """The desired frequency for each notification in the category"""
        data["notification_preferences[frequency]"] = notification_preferences_frequency

        self.logger.debug("PUT /api/v1/users/self/communication_channels/{communication_channel_id}/notification_preference_categories/{category} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/users/self/communication_channels/{communication_channel_id}/notification_preference_categories/{category}".format(**path), data=data, params=params, no_data=True)