def get_authorization_query_session_for_vault(self, vault_id):
        """Gets the ``OsidSession`` associated with the authorization query service for the given vault.

        arg:    vault_id (osid.id.Id): the ``Id`` of the vault
        return: (osid.authorization.AuthorizationQuerySession) - ``an
                _authorization_query_session``
        raise:  NotFound - ``vault_id`` not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_authorization_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization_query()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_authorization_query():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AuthorizationQuerySession(vault_id, runtime=self._runtime)