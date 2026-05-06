def get_assessment_taken_query_session_for_bank(self, bank_id):
        """Gets the ``OsidSession`` associated with the assessment taken query service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the bank
        return: (osid.assessment.AssessmentTakenQuerySession) - an
                ``AssessmentTakenQuerySession``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_assessment_taken_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_taken_query()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_assessment_taken_query():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssessmentTakenQuerySession(bank_id, runtime=self._runtime)