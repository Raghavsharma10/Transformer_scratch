def delete_user_login(self, id, user_id):
        """
        Delete a user login.

        Delete an existing login.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("DELETE /api/v1/users/{user_id}/logins/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/users/{user_id}/logins/{id}".format(**path), data=data, params=params, no_data=True)