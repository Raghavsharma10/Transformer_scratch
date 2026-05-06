def get_family_hierarchy_session(self, proxy=None):
        """Gets the ``OsidSession`` associated with the family hierarchy service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.FamilyHierarchySession) - a
                ``FamilyHierarchySession`` for families
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_family_hierarchy()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_family_hierarchy()`` is ``true``.*

        """
        if not self.supports_family_hierarchy():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.FamilyHierarchySession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session