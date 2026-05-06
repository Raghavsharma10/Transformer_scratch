def get_assessment_bank_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment banking service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentBankSession) - an
                ``AssessmentBankSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_bank()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_bank()`` is ``true``.*

        """
        if not self.supports_assessment_bank():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentBankSession(proxy=proxy, runtime=self._runtime)