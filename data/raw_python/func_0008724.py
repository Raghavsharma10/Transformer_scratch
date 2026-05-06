def has_verified_email(self):
        """
        Has the user verified that the email he has given is legit?

        Verified e-mail is required to the gallery. Confirmation happens by
        sending an email to the user and the owner of the email user verifying
        that he is the same as the Imgur user.
        """
        url = (self._imgur._base_url + "/3/account/{0}/"
               "verifyemail".format(self.name))
        return self._imgur._send_request(url, needs_auth=True)