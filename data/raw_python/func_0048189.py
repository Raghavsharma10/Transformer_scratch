def on_message(self, data):
    """ Todo connect """
    if self.user.id != data.author.id:
      self.logger.info("Message from %s: %s", data.author, data.content)
      if not isinstance(self.con_message, type(None)):
        self.logger.debug(type(self.con_message))
        self.con_message(data)