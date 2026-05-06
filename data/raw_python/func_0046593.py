def get_grade_entry_admin_session(self):
        """Gets the ``OsidSession`` associated with the grade entry administration service.

        return: (osid.grading.GradeEntryAdminSession) - a
                ``GradeEntryAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_grade_entry_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_entry_admin()`` is ``true``.*

        """
        if not self.supports_grade_entry_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeEntryAdminSession(runtime=self._runtime)