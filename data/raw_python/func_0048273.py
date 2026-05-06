def get_resource_query_session(self, proxy):
        """Gets a resource query session.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceQuerySession) - ``a
                ResourceQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_query()`` is ``true``.*

        """
        if not self.supports_resource_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceQuerySession(proxy=proxy, runtime=self._runtime)