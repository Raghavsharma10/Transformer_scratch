def get_assessment_results_session_for_bank(self, bank_id, proxy):
        """Gets an ``AssessmentResultsSession`` to retrieve assessment results for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the assessment taken
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentResultsSession) - an
                assessment results session for this service
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_results()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_results()`` is ``true``.*

        """
        if not self.supports_assessment_results():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssessmentResultsSession(bank_id, proxy, self._runtime)