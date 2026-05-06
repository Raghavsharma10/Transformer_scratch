def get_authorization_lookup_session_for_vault(self, vault_id, proxy):
        """Gets the ``OsidSession`` associated with the authorization lookup service for the given vault.

        arg:    vault_id (osid.id.Id): the ``Id`` of the vault
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.authorization.AuthorizationLookupSession) - ``an
                _authorization_lookup_session``
        raise:  NotFound - ``vault_id`` not found
        raise:  NullArgument - ``vault_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_authorization_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization_lookup()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_authorization_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AuthorizationLookupSession(vault_id, proxy, self._runtime)