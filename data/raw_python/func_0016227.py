def get_user(self, username):
        """Retrieve information about a user

        Returns:
            dict: User information

            None: If no user or failure occurred
        """

        response = self._get(self.rest_url + "/user",
                             params={"username": username,
                                     "expand": "attributes"})

        if not response.ok:
            return None

        return response.json()