def set_password(self, service, username, password):
        """Write the password to the registry
        """
        # encrypt the password
        password_encrypted = _win_crypto.encrypt(password.encode('utf-8'))
        # encode with base64
        password_base64 = base64.encodestring(password_encrypted)
        # encode again to unicode
        password_saved = password_base64.decode('ascii')

        # store the password
        key_name = self._key_for_service(service)
        hkey = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_name)
        winreg.SetValueEx(hkey, username, 0, winreg.REG_SZ, password_saved)