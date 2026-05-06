def replacePassword(self, currentPassword, newPassword):
        """
        Set this account's password if the current password matches.

        @param currentPassword: The password to match against the current one.
        @param newPassword: The new password.

        @return: A deferred firing when the password has been changed.
        @raise BadCredentials: If the current password did not match.
        """
        if unicode(currentPassword) != self.password:
            return fail(BadCredentials())
        return self.setPassword(newPassword)