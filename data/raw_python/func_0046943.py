def get_composition_repository_session(self, proxy):
        """Gets the session for retrieving composition to repository mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.repository.CompositionRepositorySession) - a
                ``CompositionRepositorySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_composition_repository()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_composition_repository()`` is ``true``.*

        """
        if not self.supports_composition_repository():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CompositionRepositorySession(proxy=proxy, runtime=self._runtime)