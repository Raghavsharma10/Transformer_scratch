def writemessage(self, text):
        """Put data in output queue, rebuild the prompt and entered data"""
        # Need to grab the input queue lock to ensure the entered data doesn't change
        # before we're done rebuilding it.
        # Note that writemessage will eventually call writecooked
        self.IQUEUELOCK.acquire()
        TelnetHandlerBase.writemessage(self, text)
        self.IQUEUELOCK.release()