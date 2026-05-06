def full_format(self):
        """
        Returns the full name (first and last parts), and the username between brackets if the user has it.
        If there is no info about the user, returns the user id between < and >.
        """
        formatted_user = self.full_name
        if self.user.username is not None:
            formatted_user += " [" + self.user.username + "]"
        if not formatted_user:
            formatted_user = self._id()
        return formatted_user