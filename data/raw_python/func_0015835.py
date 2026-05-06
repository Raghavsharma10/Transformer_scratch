def decrypt(self, password_encrypted):
        """Decrypt the password.
        """
        if not password_encrypted or not self._crypter:
            return password_encrypted or b''
        return self._crypter.decrypt(password_encrypted)