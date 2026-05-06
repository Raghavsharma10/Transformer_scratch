def get_hmac(self, key):
        """Returns the keyed HMAC for authentication of this state data.

        :param key: the key for the keyed hash function
        """
        h = HMAC.new(key, None, SHA256)
        h.update(self.iv)
        h.update(str(self.chunks).encode())
        h.update(self.f_key)
        h.update(self.alpha_key)
        h.update(str(self.encrypted).encode())
        return h.digest()