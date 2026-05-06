def get_vault_admin_session(self):
        """Gets the OsidSession associated with the vault administration service.

        return: (osid.authorization.VaultAdminSession) - a
                ``VaultAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_vault_admin() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_vault_admin()`` is true.*

        """
        if not self.supports_vault_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.VaultAdminSession(runtime=self._runtime)