def get_hierarchy_admin_session(self, proxy):
        """Gets the hierarchy administrative session.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.hierarchy.HierarchyAdminSession) - a
                ``HierarchyAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_hierarchy_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_hierarchy_admin()`` is ``true``.*

        """
        if not self.supports_hierarchy_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.HierarchyAdminSession(proxy=proxy, runtime=self._runtime)