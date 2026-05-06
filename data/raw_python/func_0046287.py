def get_assessment_taken_lookup_session(self):
        """Gets the ``OsidSession`` associated with the assessment taken lookup service.

        return: (osid.assessment.AssessmentTakenLookupSession) - an
                ``AssessmentTakenLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_taken_lookup()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_taken_lookup()`` is ``true``.*

        """
        if not self.supports_assessment_taken_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentTakenLookupSession(runtime=self._runtime)