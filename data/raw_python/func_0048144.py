def on_message(self, ws, message):
    """ Todo """
    m = json.loads(message)
    self.logger.debug(m)
    if m.get("s", 0):
      self.sequence = m["s"]
    if m["op"] == self.DISPATCH:
      if m["t"] == "READY":
        for channel in m["d"]["private_channels"]:
          if len(channel["recipients"]) == 1:
            self.channels[channel["id"]] = User(channel["recipients"][0])
            self.logger.info("added channel for %s", self.channels[channel["id"]])
        self.session = m["d"]["session_id"]
        self.con_connect(User(m["d"]["user"]))
      elif m["t"] == "GUILD_CREATE":
        pass
      elif m["t"] == "MESSAGE_CREATE":
#        if not m["d"]["channel_id"] in self.channels:
#        print("ch:")
#        print(self.get("channels/"+m["d"]["channel_id"]))
        self.con_message(Message(m["d"]))
    elif m["op"] == self.HELLO:
      interval = int(m['d']['heartbeat_interval'] / 1000)
      self.h = Heartbeat(self, interval)
      self.h.daemon = True
      self.h.start()
    elif m["op"] == self.HEARTBEAT_ACK:
      pass
    else:
      self.logger.debug(m)