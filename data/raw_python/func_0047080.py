def get_authorization_vault_assignment_session(self):
        """Gets the session for assigning authorizations to vault mappings.

        return: (osid.authorization.AuthorizationVaultAssignmentSession)
                - a ``AuthorizationVaultAssignmentSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_authorization_vault_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization_vault_assignment()`` is ``true``.*

        """
        if not self.supports_authorization_vault_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AuthorizationVaultAssignmentSession(runtime=self._runtime)