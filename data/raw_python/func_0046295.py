def get_assessment_session(self, proxy):
        """Gets an ``AssessmentSession`` which is responsible for taking assessments and examining responses from assessments taken.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentSession) - an assessment
                session for this service
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment()`` is ``true``.*

        """
        if not self.supports_assessment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentSession(proxy=proxy, runtime=self._runtime)