def list_user_logins_users(self, user_id):
        """
        List user logins.

        Given a user ID, return that user's logins for the given account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        self.logger.debug("GET /api/v1/users/{user_id}/logins with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/logins".format(**path), data=data, params=params, all_pages=True)