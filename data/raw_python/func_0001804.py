def split(self, encoded):
        """ Split into signature and message """
        maxlen = len(encoded) - self.sig_size
        message = encoded[:maxlen]
        signature = encoded[-self.sig_size:]
        return message, signature