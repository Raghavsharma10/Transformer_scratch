def get_log_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the log lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogLookupSession) - a ``LogLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_lookup()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_lookup()`` is ``true``.*

        """
        if not self.supports_log_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LogLookupSession(proxy=proxy, runtime=self._runtime)