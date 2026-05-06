def get_group_hierarchy_session(self, proxy):
        """Gets the group hierarchy traversal session for the given resource group.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.BinHierarchySession) - ``a
                GroupHierarchySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_group_hierarchy()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_group_hierarchy()`` is ``true``.*

        """
        if not self.supports_group_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BinHierarchySession(proxy=proxy, runtime=self._runtime)