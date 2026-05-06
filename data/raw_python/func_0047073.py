def get_authorization_session_for_vault(self, vault_id):
        """Gets an ``AuthorizationSession`` which is responsible for performing authorization checks for the given vault.

        arg:    vault_id (osid.id.Id): the ``Id`` of the vault
        return: (osid.authorization.AuthorizationSession) - ``an
                _authorization_session``
        raise:  NotFound - ``vault_id``
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_authorization()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_authorization():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AuthorizationSession(vault_id, runtime=self._runtime)