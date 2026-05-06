def get_bin_query_session(self, proxy):
        """Gets the bin query session.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.BinQuerySession) - a ``BinQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bin_query()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_bin_query()`` is ``true``.*

        """
        if not self.supports_bin_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BinQuerySession(proxy=proxy, runtime=self._runtime)