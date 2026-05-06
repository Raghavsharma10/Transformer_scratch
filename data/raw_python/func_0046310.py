def get_assessment_basic_authoring_session_for_bank(self, bank_id, proxy):
        """Gets the ``OsidSession`` associated with the assessment authoring service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of a bank
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentBasicAuthoringSession) - an
                ``AssessmentBasicAuthoringSession``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_assessment_basic_authoring()`` or
                ``supports_visibe_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_basic_authoring()`` and
        ``supports_visibe_federation()`` is ``true``.*

        """
        if not self.supports_assessment_basic_authoring():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssessmentBasicAuthoringSession(bank_id, proxy, self._runtime)