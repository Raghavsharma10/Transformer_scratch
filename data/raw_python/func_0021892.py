def full_data(self):
        """
        Returns all the info available for the chat in the following format:
            title [username] (type) <id>
        If any data is not available, it is not added.
        """
        data = [
            self.chat.title,
            self._username(),
            self._type(),
            self._id()
        ]
        return " ".join(filter(None, data))