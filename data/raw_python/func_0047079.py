def get_authorization_vault_session(self):
        """Gets the session for retrieving authorization to vault mappings.

        return: (osid.authorization.AuthorizationVaultSession) - an
                ``AuthorizationVaultSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_authorization_vault()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization_vault()`` is ``true``.*

        """
        if not self.supports_authorization_vault():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AuthorizationVaultSession(runtime=self._runtime)