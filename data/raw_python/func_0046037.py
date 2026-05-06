def get_relationship_query_session(self, proxy=None):
        """Gets the ``OsidSession`` associated with the relationship query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.RelationshipQuerySession) - a
                ``RelationshipQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_query()`` is ``true``.*

        """
        if not self.supports_relationship_query():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.RelationshipQuerySession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session