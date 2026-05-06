def get_session(self, username, password, remote="127.0.0.1", proxy=None):
        """Create a session for a user.

        Attempts to create a user session on the Crowd server.

        Args:
            username: The account username.

            password: The account password.

            remote:
                The remote address of the user. This can be used
                to create multiple concurrent sessions for a user.
                The host you run this program on may need to be configured
                in Crowd as a trusted proxy for this to work.

            proxy: Value of X-Forwarded-For server header.

        Returns:
            dict:
                A dict mapping of user attributes if the application
                authentication was successful. See the Crowd
                documentation for the authoritative list of attributes.

            None: If authentication failed.
        """

        params = {
            "username": username,
            "password": password,
            "validation-factors": {
                "validationFactors": [
                    {"name": "remote_address", "value": remote, },
                ]
            }
        }

        if proxy:
            params["validation-factors"]["validationFactors"].append({"name": "X-Forwarded-For", "value": proxy, })

        response = self._post(self.rest_url + "/session",
                              data=json.dumps(params),
                              params={"expand": "user"})

        # If authentication failed for any reason return None
        if not response.ok:
            return None

        # Otherwise return the user object
        return response.json()