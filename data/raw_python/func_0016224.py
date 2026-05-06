def validate_session(self, token, remote="127.0.0.1", proxy=None):
        """Validate a session token.

        Validate a previously acquired session token against the
        Crowd server. This may be a token provided by a user from
        a http cookie or by some other means.

        Args:
            token: The session token.

            remote: The remote address of the user.

            proxy: Value of X-Forwarded-For server header

        Returns:
            dict:
                A dict mapping of user attributes if the application
                authentication was successful. See the Crowd
                documentation for the authoritative list of attributes.

            None: If authentication failed.
        """

        params = {
            "validationFactors": [
                {"name": "remote_address", "value": remote, },
            ]
        }

        if proxy:
            params["validation-factors"]["validationFactors"].append({"name": "X-Forwarded-For", "value": proxy, })

        url = self.rest_url + "/session/%s" % token
        response = self._post(url, data=json.dumps(params), params={"expand": "user"})

        # For consistency between methods use None rather than False
        # If token validation failed for any reason return None
        if not response.ok:
            return None

        # Otherwise return the user object
        return response.json()