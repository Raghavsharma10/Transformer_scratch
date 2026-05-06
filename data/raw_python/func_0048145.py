def on_error(self, ws, error):
    """ Todo """
    if type(error).__name__ == "KeyboardInterrupt":
      sys.exit()
    self.logger.debug("error")