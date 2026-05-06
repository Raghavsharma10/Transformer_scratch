def get_bin_lookup_session(self, proxy):
        """Gets the bin lookup session.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.BinLookupSession) - a
                ``BinLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bin_lookup()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_bin_lookup()`` is ``true``.*

        """
        if not self.supports_bin_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BinLookupSession(proxy=proxy, runtime=self._runtime)