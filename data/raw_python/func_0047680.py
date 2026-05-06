def get_log_entry_log_session(self, proxy):
        """Gets the session for retrieving log entry to log mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogEntryLogSession) - a
                ``LogEntryLogSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_entry_log()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_entry_log()`` is ``true``.*

        """
        if not self.supports_log_entry_log():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LogEntryLogSession(proxy=proxy, runtime=self._runtime)