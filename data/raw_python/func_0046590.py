def get_grade_system_admin_session(self):
        """Gets the ``OsidSession`` associated with the grade system administration service.

        return: (osid.grading.GradeSystemAdminSession) - a
                ``GradeSystemAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_grade_system_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_system_admin()`` is ``true``.*

        """
        if not self.supports_grade_system_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeSystemAdminSession(runtime=self._runtime)