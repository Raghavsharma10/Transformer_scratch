def get_resource_admin_session(self, proxy):
        """Gets a resource administration session for creating, updating and deleting resources.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceAdminSession) - ``a
                ResourceAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_admin()`` is ``true``.*

        """
        if not self.supports_resource_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceAdminSession(proxy=proxy, runtime=self._runtime)