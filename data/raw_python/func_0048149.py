def send_heartbeat(self):
    """ Todo """
    self.logger.debug("heartbeat "+str(self.t.sequence))
    self.t.ws.send(json.dumps({'op': self.t.HEARTBEAT,
                               'd': self.t.sequence}))