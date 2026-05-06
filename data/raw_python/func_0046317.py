def get_assessment_taken_bank_session(self, proxy):
        """Gets the session for retrieving taken assessments to bank mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentTakenBankSession) - an
                ``AssessmentTakenBankSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_taken_bank()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_taken_bank()`` is ``true``.*

        """
        if not self.supports_assessment_taken_bank():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentTakenBankSession(proxy=proxy, runtime=self._runtime)