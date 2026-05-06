def send_message(self, user=None, message=None, channel=None):
    """ Todo """
    self.logger.info("sending message to %s: %s", user, message)
    cid=channel
    if not cid:
      for cid in self.channels:
        if str(self.channels[cid]) == str(user):
          channel=cid
          self.logger.debug(cid)
    if (channel):
      self.post('channels/'+cid+'/messages',
                json.dumps({'content': message,
                            'nonce': random_integer(-2**63, 2**63 - 1)}))
    else:
      logger.error("Unknown user %s",user)