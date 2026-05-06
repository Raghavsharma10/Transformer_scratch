def send_message(self, user=None, message=None,channel=None):
    """ Todo connect """
    self.transport.send_message(user=user, message=message, channel=channel)