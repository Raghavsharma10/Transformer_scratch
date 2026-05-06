def full_data(self):
        """
        Returns all the info available for the user in the following format:
            name [username] <id> (locale) bot_or_user
        If any data is not available, it is not added.
        """
        data = [
            self.full_name,
            self._username(),
            self._id(),
            self._language_code(),
            self._is_bot()
        ]
        return " ".join(filter(None, data))