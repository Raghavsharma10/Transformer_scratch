def get_assessment_bank_assignment_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment bank assignment service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentBankAssignmentSession) - an
                ``AssessmentBankAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_assessment_bank_assignment()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_bank_assignment()`` is ``true``.*

        """
        if not self.supports_assessment_bank_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentBankAssignmentSession(proxy=proxy, runtime=self._runtime)