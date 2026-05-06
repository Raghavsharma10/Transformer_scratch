def terminate_session(self, token):
        """Terminates the session token, effectively logging out the user
        from all crowd-enabled services.

        Args:
            token: The session token.

        Returns:
            True: If session terminated

            None: If session termination failed
        """

        url = self.rest_url + "/session/%s" % token
        response = self._delete(url)

        # For consistency between methods use None rather than False
        # If token validation failed for any reason return None
        if not response.ok:
            return None

        # Otherwise return True
        return True