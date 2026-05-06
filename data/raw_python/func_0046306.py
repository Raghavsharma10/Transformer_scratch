def get_assessment_query_session_for_bank(self, bank_id, proxy):
        """Gets the ``OsidSession`` associated with the assessment query service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the bank
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentQuerySession) - ``an
                _assessment_query_session``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_assessment_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_query()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_assessment_query():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssessmentQuerySession(bank_id, proxy, self._runtime)