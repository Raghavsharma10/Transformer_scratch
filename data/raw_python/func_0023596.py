def send(self, command, message=None):
        '''Send a command over the socket with length endcoded'''
        if message:
            joined = command + constants.NL + util.pack(message)
        else:
            joined = command + constants.NL
        if self._blocking:
            for sock in self.socket():
                sock.sendall(joined)
        else:
            self._pending.append(joined)