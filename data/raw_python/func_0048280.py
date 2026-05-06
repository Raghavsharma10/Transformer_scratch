def get_resource_bin_session(self, proxy):
        """Gets the session for retrieving resource to bin mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceBinSession) - a
                ``ResourceBinSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_bin()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_bin()`` is ``true``.*

        """
        if not self.supports_resource_bin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceBinSession(proxy=proxy, runtime=self._runtime)