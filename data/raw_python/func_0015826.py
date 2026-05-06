def delete_password(self, service, username):
        """Delete the password for the username of the service.
        """
        items = self._find_passwords(service, username, deleting=True)
        if not items:
            raise PasswordDeleteError("Password not found")
        for current in items:
            result = GnomeKeyring.item_delete_sync(current.keyring,
                                                   current.item_id)
            if result == GnomeKeyring.Result.CANCELLED:
                raise PasswordDeleteError("Cancelled by user")
            elif result != GnomeKeyring.Result.OK:
                raise PasswordDeleteError(result.value_name)