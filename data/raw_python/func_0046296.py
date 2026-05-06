def get_assessment_session_for_bank(self, bank_id, proxy):
        """Gets an ``AssessmentSession`` which is responsible for performing assessments for the given bank ``Id``.

        arg:    bank_id (osid.id.Id): the ``Id`` of a bank
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentSession) - an assessment
                session for this service
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment()`` is ``true``.*

        """
        if not self.supports_assessment():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssessmentSession(bank_id, proxy, self._runtime)