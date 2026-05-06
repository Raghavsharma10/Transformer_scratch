def connect(self):
    """ Todo connect """
    self.transport = Transport(self.token, on_connect=self.on_connect, on_message=self.on_message)