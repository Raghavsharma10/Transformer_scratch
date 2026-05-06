def get_hmac(self, key):
        """Returns the keyed HMAC for authentication of this state data.

        :param key: the key for the keyed hash function
        """
        h = HMAC.new(key, None, SHA256)
        h.update(str(self.index).encode())
        h.update(self.seed)
        h.update(str(self.n).encode())
        h.update(self.root)
        h.update(str(self.timestamp).encode())
        return h.digest()