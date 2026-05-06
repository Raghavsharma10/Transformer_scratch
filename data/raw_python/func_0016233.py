def send_password_reset_link(self, username):
        """Sends the user a password reset link (by email)

        Args:
            username: The account username.

        Returns:
            True: Succeeded
            False: If unsuccessful
        """

        response = self._post(self.rest_url + "/user/mail/password",
                              params={"username": username})

        if response.ok:
            return True

        return False