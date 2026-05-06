def get_assessment_offered_lookup_session(self):
        """Gets the ``OsidSession`` associated with the assessment offered lookup service.

        return: (osid.assessment.AssessmentOfferedLookupSession) - an
                ``AssessmentOfferedLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_offered_lookup()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_offered_lookup()`` is ``true``.*

        """
        if not self.supports_assessment_offered_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentOfferedLookupSession(runtime=self._runtime)