def tag(self, value):
        """The name of the program that generated the log message.

        The tag can only contain alphanumeric
        characters. If the tag is longer than {MAX_TAG_LEN} characters
        it will be truncated automatically.

        """
        if value is None:
            value = sys.argv[0]
        self._tag = value[:self.MAX_TAG_LEN]