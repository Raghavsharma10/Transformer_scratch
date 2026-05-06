def get_hierarchy_traversal_session(self, proxy):
        """Gets the ``OsidSession`` associated with the hierarchy traversal service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.hierarchy.HierarchyTraversalSession) - a
                ``HierarchyTraversalSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_hierarchy_traversal()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_hierarchy_traversal()`` is ``true``.*

        """
        if not self.supports_hierarchy_traversal():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.HierarchyTraversalSession(proxy=proxy, runtime=self._runtime)