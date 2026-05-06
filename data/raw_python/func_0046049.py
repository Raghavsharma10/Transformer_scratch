def get_family_hierarchy_design_session(self, proxy=None):
        """Gets the ``OsidSession`` associated with the family hierarchy design service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.FamilyHierarchyDesignSession) - a
                ``HierarchyDesignSession`` for families
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_family_hierarchy_design()``
                is ``false``
        *compliance: optional -- This method must be implemented if ``supports_family_hierarchy_design()`` is ``true``.*

        """
        if not self.supports_family_hierarchy_design():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.FamilyHierarchyDesignSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session