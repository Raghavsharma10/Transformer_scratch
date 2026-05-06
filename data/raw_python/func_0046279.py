def get_assessment_admin_session(self):
        """Gets the ``OsidSession`` associated with the assessment administration service.

        return: (osid.assessment.AssessmentAdminSession) - an
                ``AssessmentAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_admin()`` is ``true``.*

        """
        if not self.supports_assessment_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentAdminSession(runtime=self._runtime)