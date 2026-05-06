def get_assessment_offered_bank_assignment_session(self, proxy):
        """Gets the session for assigning offered assessments to bank mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentOfferedBankAssignmentSession)
                - an ``AssessmentOfferedBankAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_assessment_offered_bank_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_offered_bank_assignment()`` is ``true``.*

        """
        if not self.supports_assessment_offered_bank_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentOfferedBankAssignmentSession(proxy=proxy, runtime=self._runtime)