def get_log_entry_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the logging reading service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogEntryLookupSession) - a
                ``LogEntryLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_entry_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_entry_lookup()`` is ``true``.*

        """
        if not self.supports_log_entry_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LogEntryLookupSession(proxy=proxy, runtime=self._runtime)