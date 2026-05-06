def get_password(self, service, username):
        """Get password of the username for the service
        """
        try:
            # fetch the password
            key = self._key_for_service(service)
            hkey = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
            password_saved = winreg.QueryValueEx(hkey, username)[0]
            password_base64 = password_saved.encode('ascii')
            # decode with base64
            password_encrypted = base64.decodestring(password_base64)
            # decrypted the password
            password = _win_crypto.decrypt(password_encrypted).decode('utf-8')
        except EnvironmentError:
            password = None
        return password