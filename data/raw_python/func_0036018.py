def encrypt(self, key):
        """This method encrypts and signs the state to make it unreadable by
        the server, since it contains information that would allow faking
        proof of storage.

        :param key: the key to encrypt and sign with
        """
        if (self.encrypted):
            return
        # encrypt
        self.iv = Random.new().read(AES.block_size)
        aes = AES.new(key, AES.MODE_CFB, self.iv)
        self.f_key = aes.encrypt(self.f_key)
        self.alpha_key = aes.encrypt(self.alpha_key)
        self.encrypted = True
        # sign
        self.hmac = self.get_hmac(key)