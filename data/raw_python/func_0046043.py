def get_relationship_family_session(self, proxy=None):
        """Gets the ``OsidSession`` to lookup relationship/family mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.RelationshipFamilySession) - a
                ``RelationshipFamilySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_family()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_family()``
            is ``true``.*

        """
        if not self.supports_relationship_family():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.RelationshipFamilySession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session