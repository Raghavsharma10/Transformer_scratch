def message(self, subject, text):
        """Compose a message to this user.  Calls :meth:`narwal.Reddit.compose`.
        
        :param subject: subject of message
        :param text: body of message
        """
        return self._reddit.compose(self.name, subject, text)