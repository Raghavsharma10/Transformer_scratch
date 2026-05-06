def writecooked(self, text):
        """Put data directly into the output queue"""
        # Ensure this is the only thread writing
        self.OQUEUELOCK.acquire()
        TelnetHandlerBase.writecooked(self, text)
        self.OQUEUELOCK.release()