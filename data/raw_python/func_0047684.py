def get_log_hierarchy_session(self, proxy):
        """Gets the ``OsidSession`` associated with the log hierarchy service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogHierarchySession) - a
                ``LogHierarchySession`` for logs
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_hierarchy()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_hierarchy()`` is ``true``.*

        """
        if not self.supports_log_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LogHierarchySession(proxy=proxy, runtime=self._runtime)