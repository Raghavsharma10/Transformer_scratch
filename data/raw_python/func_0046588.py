def get_grade_system_query_session(self):
        """Gets the ``OsidSession`` associated with the grade system query service.

        return: (osid.grading.GradeSystemQuerySession) - a
                ``GradeSystemQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_grade_system_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_system_query()`` is ``true``.*

        """
        if not self.supports_grade_system_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeSystemQuerySession(runtime=self._runtime)