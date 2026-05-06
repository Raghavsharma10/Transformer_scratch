def get_assessment_offered_query_session(self):
        """Gets the ``OsidSession`` associated with the assessment offered query service.

        return: (osid.assessment.AssessmentOfferedQuerySession) - an
                ``AssessmentOfferedQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_offered_query()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_offered_query()`` is ``true``.*

        """
        if not self.supports_assessment_offered_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentOfferedQuerySession(runtime=self._runtime)