def close(self):
        """Closes the ssh connection."""
        if 'isLive' in self.__dict__ and self.isLive:
            self.transport.close()
            self.isLive = False