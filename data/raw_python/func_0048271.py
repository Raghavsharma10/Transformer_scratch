def get_resource_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the resource lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceLookupSession) - ``a
                ResourceLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_lookup()`` is ``true``.*

        """
        if not self.supports_resource_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceLookupSession(proxy=proxy, runtime=self._runtime)