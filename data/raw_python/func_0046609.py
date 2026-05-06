def get_gradebook_column_gradebook_assignment_session(self, proxy):
        """Gets the session for assigning gradebook column to gradebook mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradebookColumnGradebookAssignmentSession)
                - a ``GradebookColumnGradebookAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_gradebook_column_gradebook_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_column_gradebook_assignment()`` is
        ``true``.*

        """
        if not self.supports_gradebook_column_gradebook_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookColumnGradebookAssignmentSession(proxy=proxy, runtime=self._runtime)