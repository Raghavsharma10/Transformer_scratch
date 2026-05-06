def get_cipher(self):
        """
        Return a new Cipher object for each time we want to encrypt/decrypt. This is because
        pgcrypto expects a zeroed block for IV (initial value), but the IV on the cipher
        object is cumulatively updated each time encrypt/decrypt is called.
        """
        return self.cipher_class.new(self.cipher_key, self.cipher_class.MODE_CBC, b'\0' * self.cipher_class.block_size)