def get_gradebook_lookup_session(self, proxy):
        """Gets the OsidSession associated with the gradebook lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradebookLookupSession) - a
                ``GradebookLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_gradebook_lookup() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_lookup()`` is true.*

        """
        if not self.supports_gradebook_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookLookupSession(proxy=proxy, runtime=self._runtime)