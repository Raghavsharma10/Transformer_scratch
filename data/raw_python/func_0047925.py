def get_hierarchy_design_session(self, proxy):
        """Gets the ``OsidSession`` associated with the hierarchy design service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.hierarchy.HierarchyDesignSession) - a
                ``HierarchyDesignSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_hierarchy_design()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_hierarchy_design()`` is ``true``.*

        """
        if not self.supports_hierarchy_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.HierarchyDesignSession(proxy=proxy, runtime=self._runtime)