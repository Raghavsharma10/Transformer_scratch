def listen(self):
        """
        Listen/Connect to message service loop to start receiving messages.
        Do not include in constructor, in this way it can be included in tasks
        """
        self.listening = True
        try:
            self.service_backend.listen()
        except AuthenticationError:
            self.listening = False
            raise
        else:
            self.listening = False