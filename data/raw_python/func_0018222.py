def setup(self):
        "Connect incoming connection to a telnet session"
        try:
            self.TERM = self.request.term
        except:
            pass
        self.setterm(self.TERM)
        self.sock = self.request._sock
        for k in self.DOACK.keys():
            self.sendcommand(self.DOACK[k], k)
        for k in self.WILLACK.keys():
            self.sendcommand(self.WILLACK[k], k)