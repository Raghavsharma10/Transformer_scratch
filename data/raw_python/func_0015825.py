def set_password(self, service, username, password):
        """Set password for the username of the service
        """
        service = self._safe_string(service)
        username = self._safe_string(username)
        password = self._safe_string(password)
        attrs = GnomeKeyring.Attribute.list_new()
        GnomeKeyring.Attribute.list_append_string(attrs, 'username', username)
        GnomeKeyring.Attribute.list_append_string(attrs, 'service', service)
        GnomeKeyring.Attribute.list_append_string(
            attrs, 'application', 'python-keyring')
        result = GnomeKeyring.item_create_sync(
            self.keyring_name, GnomeKeyring.ItemType.NETWORK_PASSWORD,
            "Password for '%s' on '%s'" % (username, service),
            attrs, password, True)[0]
        if result == GnomeKeyring.Result.CANCELLED:
            # The user pressed "Cancel" when prompted to unlock their keyring.
            raise PasswordSetError("Cancelled by user")
        elif result != GnomeKeyring.Result.OK:
            raise PasswordSetError(result.value_name)