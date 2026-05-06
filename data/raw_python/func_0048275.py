def get_resource_search_session(self, proxy):
        """Gets a resource search session.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceSearchSession) - ``a
                ResourceSearchSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_search()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_search()`` is ``true``.*

        """
        if not self.supports_resource_search():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceSearchSession(proxy=proxy, runtime=self._runtime)