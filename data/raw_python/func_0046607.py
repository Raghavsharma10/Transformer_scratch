def get_gradebook_column_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the gradebook column query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradebookColumnQuerySession) - a
                ``GradebookColumnQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_gradebook_column_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_column_query()`` is ``true``.*

        """
        if not self.supports_gradebook_column_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookColumnQuerySession(proxy=proxy, runtime=self._runtime)