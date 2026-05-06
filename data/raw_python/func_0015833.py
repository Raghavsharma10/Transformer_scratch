def delete_password(self, service, username):
        """Delete the password for the username of the service.
        """
        try:
            key_name = self._key_for_service(service)
            hkey = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, key_name, 0,
                winreg.KEY_ALL_ACCESS)
            winreg.DeleteValue(hkey, username)
            winreg.CloseKey(hkey)
        except WindowsError:
            e = sys.exc_info()[1]
            raise PasswordDeleteError(e)
        self._delete_key_if_empty(service)