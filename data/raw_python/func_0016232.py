def change_password(self, username, newpassword, raise_on_error=False):
        """Change new password for a user

        Args:
            username: The account username.

            newpassword: The account new password.

            raise_on_error: optional (default: False)

        Returns:
            True: Succeeded
            False: If unsuccessful
        """

        response = self._put(self.rest_url + "/user/password",
                             data=json.dumps({"value": newpassword}),
                             params={"username": username})

        if response.ok:
            return True

        if raise_on_error:
            raise RuntimeError(response.json()['message'])

        return False