def disconnect(self):
        """
        Closes connection to Scratch
        """
        try: # connection may already be disconnected, so catch exceptions
            self.socket.shutdown(socket.SHUT_RDWR) # a proper disconnect
        except socket.error:
            pass
        self.socket.close()
        self.connected = False