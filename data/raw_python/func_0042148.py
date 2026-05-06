def write_message(self, status=messages.INFO, message=None):
        """
        Writes a message to django's messaging framework and
        returns the written message.

        :param status: The message status level. Defaults to \
        messages.INFO.
        :param message: The message to write. If not given, \
        defaults to appending 'saved' to the unicode representation \
        of `self.object`.
        """
        if not message:
                message = u"%s saved" % self.object
        messages.add_message(self.request, status, message)
        return message