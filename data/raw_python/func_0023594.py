def setblocking(self, blocking):
        '''Set whether or not this message is blocking'''
        for sock in self.socket():
            sock.setblocking(blocking)
            self._blocking = blocking