def _obtain_api_token(self, username, password):
        """Use username and password to obtain and return an API token."""
        data = self._request("POST", "auth/apitoken",
                             {"username": username, "password": password},
                             reestablish_session=False)
        return data["api_token"]