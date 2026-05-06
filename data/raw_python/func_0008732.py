def send_verification_email(self):
        """
        Send verification email to this users email address.

        Remember that the verification email may end up in the users spam
        folder.
        """
        url = (self._imgur._base_url + "/3/account/{0}"
               "/verifyemail".format(self.name))
        self._imgur._send_request(url, needs_auth=True, method='POST')