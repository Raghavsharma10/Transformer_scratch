def get_assessment_taken_lookup_session_for_bank(self, bank_id):
        """Gets the ``OsidSession`` associated with the assessment taken lookup service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the bank
        return: (osid.assessment.AssessmentTakenLookupSession) - an
                ``AssessmentTakenLookupSession``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_assessment_taken_lookup()``
                or ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_taken_lookup()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_assessment_taken_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssessmentTakenLookupSession(bank_id, runtime=self._runtime)