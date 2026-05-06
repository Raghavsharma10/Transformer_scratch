def get_log_entry_log_assignment_session(self, proxy):
        """Gets the session for assigning log entry to log mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogEntryLogAssignmentSession) - a
                ``LogEntryLogAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_entry_log_assignment()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_entry_log_assignment()`` is ``true``.*

        """
        if not self.supports_log_entry_log_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LogEntryLogAssignmentSession(proxy=proxy, runtime=self._runtime)