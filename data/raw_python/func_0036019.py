def decrypt(self, key):
        """This method checks the signature on the state and decrypts it.

        :param key: the key to decrypt and sign with
        """
        # check signature
        if (self.get_hmac(key) != self.hmac):
            raise HeartbeatError("Signature invalid on state.")
        if (not self.encrypted):
            return
        # decrypt
        aes = AES.new(key, AES.MODE_CFB, self.iv)
        self.f_key = aes.decrypt(self.f_key)
        self.alpha_key = aes.decrypt(self.alpha_key)
        self.encrypted = False
        self.hmac = self.get_hmac(key)