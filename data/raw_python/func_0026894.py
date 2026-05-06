def on_nicknameinuse(self, connection, event):
        """
        Increment a digit on the nickname if it's in use, and
        re-connect.
        """
        digits = ""
        while self.nickname[-1].isdigit():
            digits = self.nickname[-1] + digits
            self.nickname = self.nickname[:-1]
        digits = 1 if not digits else int(digits) + 1
        self.nickname += str(digits)
        self.connect(self.host, self.port, self.nickname)