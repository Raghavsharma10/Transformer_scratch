def connect(self):
    """ Todo """
    r = self.get('gateway')
    self.ws = websocket.WebSocketApp(r["url"]+"/?encoding=json&v=6",
                                     on_message=self.on_message,
                                     on_error=self.on_error,
                                     on_close=self.on_close)
    self.ws.on_open = self.on_connect