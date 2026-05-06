def get_grade_system_gradebook_session(self, proxy):
        """Gets the session for retrieving grade system to gradebook mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradeSystemGradebookSession) - a
                ``GradeSystemGradebookSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_grade_system_gradebook()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_system_gradebook()`` is ``true``.*

        """
        if not self.supports_grade_system_gradebook():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeSystemGradebookSession(proxy=proxy, runtime=self._runtime)