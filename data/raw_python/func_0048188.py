def on_connect(self, user):
    """ Todo connect """
    self.user = user
    self.logger.info("connected as %s", user)
    if not isinstance(self.con_connect, type(None)):
      self.con_connect(user)