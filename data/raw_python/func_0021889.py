def full_name(self):
        """
        Returns the first and last name of the user separated by a space.
        """
        formatted_user = []
        if self.user.first_name is not None:
            formatted_user.append(self.user.first_name)
        if self.user.last_name is not None:
            formatted_user.append(self.user.last_name)
        return " ".join(formatted_user)