def encrypt(self, password):
        """Encrypt the password.
        """
        if not password or not self._crypter:
            return password or b''
        return self._crypter.encrypt(password)