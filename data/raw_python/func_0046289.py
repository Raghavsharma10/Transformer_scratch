def get_assessment_taken_query_session(self):
        """Gets the ``OsidSession`` associated with the assessment taken query service.

        return: (osid.assessment.AssessmentTakenQuerySession) - an
                ``AssessmentTakenQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_taken_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_taken_query()`` is ``true``.*

        """
        if not self.supports_assessment_taken_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentTakenQuerySession(runtime=self._runtime)