def get_relationship_search_session(self, proxy=None):
        """Gets the ``OsidSession`` associated with the relationship search service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.RelationshipSearchSession) - a
                ``RelationshipSearchSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_search()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_search()`` is ``true``.*

        """
        if not self.supports_relationship_search():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.RelationshipSearchSession(proxy=proxy,
                                                         runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session