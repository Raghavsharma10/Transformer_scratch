def run(self):
    """ Todo """
    self.logger.debug("heartbeat started")
    while True:
      time.sleep(self.interval)
      self.send_heartbeat()