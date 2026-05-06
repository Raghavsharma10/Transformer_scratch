def cipher(self):
        """Retrieve information about the current cipher

        Return a triple consisting of cipher name, SSL protocol version defining
        its use, and the number of secret bits. Return None if handshaking
        has not been completed.
        """

        if not self._handshake_done:
            return

        current_cipher = SSL_get_current_cipher(self._ssl.value)
        cipher_name = SSL_CIPHER_get_name(current_cipher)
        cipher_version = SSL_CIPHER_get_version(current_cipher)
        cipher_bits = SSL_CIPHER_get_bits(current_cipher)
        return cipher_name, cipher_version, cipher_bits