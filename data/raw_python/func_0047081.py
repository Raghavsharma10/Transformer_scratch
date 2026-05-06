def get_vault_lookup_session(self):
        """Gets the OsidSession associated with the vault lookup service.

        return: (osid.authorization.VaultLookupSession) - a
                ``VaultLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_vault_lookup() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_vault_lookup()`` is true.*

        """
        if not self.supports_vault_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.VaultLookupSession(runtime=self._runtime)