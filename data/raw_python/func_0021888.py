def default_format(self):
        """
        Returns full name (first and last) if name is available.
        If not, returns username if available.
        If not available too, returns the user id as a string.
        """
        user = self.user
        if user.first_name is not None:
            return self.full_name
        elif user.username is not None:
            return user.username
        else:
            return str(user.id)