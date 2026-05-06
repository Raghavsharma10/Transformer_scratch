def get_relationship_admin_session(self, proxy=None):
        """Gets the ``OsidSession`` associated with the relationship administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.RelationshipAdminSession) - a
                ``RelationshipAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_admin()`` is ``true``.*

        """
        if not self.supports_relationship_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.RelationshipAdminSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session