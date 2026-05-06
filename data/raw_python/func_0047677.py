def get_log_entry_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the logging entry query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogEntryQuerySession) - a
                ``LogEntryQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_entry_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_entry_query()`` is ``true``.*

        """
        if not self.supports_log_entry_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LogEntryQuerySession(proxy=proxy, runtime=self._runtime)