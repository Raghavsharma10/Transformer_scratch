def user_exists(self, username):
        """Determines if the user exists.

        Args:
            username: The user name.


        Returns:
            bool:
                True if the user exists in the Crowd application.
        """

        response = self._get(self.rest_url + "/user",
                             params={"username": username})

        if not response.ok:
            return None

        return True