def content(self, value):
        """The main component of the log message.

        The content field is a freeform field that
        often begins with the process ID (pid) of the
        program that created the message.

        """
        value = self._prepend_seperator(value)
        self._content = value