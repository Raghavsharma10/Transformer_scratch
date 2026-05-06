def handleRequestForUser(self, username, url):
        """
        User C{username} wants to reset their password.  Create an attempt
        item, and send them an email if the username is valid
        """
        attempt = self.newAttemptForUser(username)
        account = self.accountByAddress(username)
        if account is None:
            # do we want to disclose this to the user?
            return
        email = self.getExternalEmail(account)
        if email is not None:
            self.sendEmail(url, attempt, email)