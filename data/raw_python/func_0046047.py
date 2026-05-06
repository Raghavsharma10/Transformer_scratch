def get_family_search_session(self, proxy=None):
        """Gets the ``OsidSession`` associated with the family search service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.FamilySearchSession) - a
                ``FamilySearchSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_family_search()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_family_search()`` is ``true``.*

        """
        if not self.supports_family_search():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.FamilySearchSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session