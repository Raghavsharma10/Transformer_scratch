def _find_passwords(self, service, username, deleting=False):
        """Get password of the username for the service
        """
        passwords = []

        service = self._safe_string(service)
        username = self._safe_string(username)
        for attrs_tuple in (('username', 'service'), ('user', 'domain')):
            attrs = GnomeKeyring.Attribute.list_new()
            GnomeKeyring.Attribute.list_append_string(
                attrs, attrs_tuple[0], username)
            GnomeKeyring.Attribute.list_append_string(
                attrs, attrs_tuple[1], service)
            result, items = GnomeKeyring.find_items_sync(
                GnomeKeyring.ItemType.NETWORK_PASSWORD, attrs)
            if result == GnomeKeyring.Result.OK:
                passwords += items
            elif deleting:
                if result == GnomeKeyring.Result.CANCELLED:
                    raise PasswordDeleteError("Cancelled by user")
                elif result != GnomeKeyring.Result.NO_MATCH:
                    raise PasswordDeleteError(result.value_name)
        return passwords