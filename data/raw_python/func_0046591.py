def get_grade_entry_query_session(self):
        """Gets the ``OsidSession`` associated with the grade entry query service.

        return: (osid.grading.GradeEntryQuerySession) - a
                ``GradeEntryQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_grade_entry_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_entry_query()`` is ``true``.*

        """
        if not self.supports_grade_entry_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeEntryQuerySession(runtime=self._runtime)