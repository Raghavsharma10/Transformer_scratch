def get_log_entry_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the logging entry administrative service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogEntryAdminSession) - a
                ``LogEntryAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_entry_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_entry_admin()`` is ``true``.*

        """
        if not self.supports_log_entry_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LogEntryAdminSession(proxy=proxy, runtime=self._runtime)