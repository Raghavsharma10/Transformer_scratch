def get_vault_query_session(self, proxy):
        """Gets the OsidSession associated with the vault query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.authorization.VaultQuerySession) - a
                ``VaultQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_vault_query() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_vault_query()`` is true.*

        """
        if not self.supports_vault_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.VaultQuerySession(proxy=proxy, runtime=self._runtime)