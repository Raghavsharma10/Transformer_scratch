def get_user_profile(self, user_id):
        """
        Get user profile.

        Returns user profile data, including user id, name, and profile pic.
        
        When requesting the profile for the user accessing the API, the user's
        calendar feed URL will be returned as well.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        self.logger.debug("GET /api/v1/users/{user_id}/profile with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/profile".format(**path), data=data, params=params, single_item=True)