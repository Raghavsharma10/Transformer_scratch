def auth_user(self, username, password):
        """Authenticate a user account against the Crowd server.

        Attempts to authenticate the user against the Crowd server.

        Args:
            username: The account username.

            password: The account password.

        Returns:
            dict:
                A dict mapping of user attributes if the application
                authentication was successful. See the Crowd documentation
                for the authoritative list of attributes.

            None: If authentication failed.
        """

        response = self._post(self.rest_url + "/authentication",
                              data=json.dumps({"value": password}),
                              params={"username": username})

        # If authentication failed for any reason return None
        if not response.ok:
            return None

        # ...otherwise return a dictionary of user attributes
        return response.json()