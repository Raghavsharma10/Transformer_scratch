def get_grade_system_gradebook_assignment_session(self, proxy):
        """Gets the session for assigning grade system to gradebook mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradeSystemGradebookSession) - a
                ``GradeSystemGradebookAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_grade_system_gradebook_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_system_gradebook_assignment()`` is ``true``.*

        """
        if not self.supports_grade_system_gradebook_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeSystemGradebookAssignmentSession(proxy=proxy, runtime=self._runtime)