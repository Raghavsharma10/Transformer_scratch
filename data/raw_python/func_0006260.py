def write_human(self, buffer_):
        """ Emulates human typing speed """

        if self.IAC in buffer_:
            buffer_ = buffer_.replace(self.IAC, self.IAC + self.IAC)
        self.msg("send %r", buffer_)
        for char in buffer_:
            delta = random.gauss(80, 20)
            self.sock.sendall(char)
            time.sleep(delta / 1000.0)