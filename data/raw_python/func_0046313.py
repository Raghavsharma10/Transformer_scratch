def get_assessment_offered_bank_session(self, proxy):
        """Gets the session for retrieving offered assessments to bank mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentOfferedBankSession) - an
                ``AssessmentOfferedBankSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_offered_bank()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_offered_bank()`` is ``true``.*

        """
        if not self.supports_assessment_offered_bank():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentOfferedBankSession(proxy=proxy, runtime=self._runtime)