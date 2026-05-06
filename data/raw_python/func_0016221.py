def auth_ping(self):
        """Test that application can authenticate to Crowd.

        Attempts to authenticate the application user against
        the Crowd server. In order for user authentication to
        work, an application must be able to authenticate.

        Returns:
            bool:
                True if the application authentication succeeded.
        """

        url = self.rest_url + "/non-existent/location"
        response = self._get(url)

        if response.status_code == 401:
            return False
        elif response.status_code == 404:
            return True
        else:
            # An error encountered - problem with the Crowd server?
            return False