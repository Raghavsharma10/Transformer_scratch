def encrypt(self, raw):
        """
        Encrypts raw data using AES and then base64 encodes it.
        :param raw:
        :return:
        """
        padded = AESCipher.pad(raw)
        init_vec = Random.new().read(AES.block_size)
        cipher = AES.new(self._key, AES.MODE_CBC, init_vec)
        return b64encode(init_vec + cipher.encrypt(padded))