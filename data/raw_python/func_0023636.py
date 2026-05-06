def sendall(self, data, flags=0):
        '''Same as socket.sendall'''
        count = len(data)
        while count:
            sent = self.send(data, flags)
            # This could probably be a buffer object
            data = data[sent:]
            count -= sent