def decrypt(self, encrypted):
        """
        Base64 decodes the data and then decrypts using AES.
        :param encrypted:
        :return:
        """
        decoded = b64decode(encrypted)
        init_vec = decoded[:AES.block_size]
        cipher = AES.new(self._key, AES.MODE_CBC, init_vec)

        return AESCipher.unpad(cipher.decrypt(decoded[AES.block_size:]))