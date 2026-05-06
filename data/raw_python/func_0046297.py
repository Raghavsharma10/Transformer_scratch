def get_assessment_results_session(self, proxy):
        """Gets an ``AssessmentResultsSession`` to retrieve assessment results.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentResultsSession) - an
                assessment results session for this service
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_results()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_results()`` is ``true``.*

        """
        if not self.supports_assessment_results():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentResultsSession(proxy=proxy, runtime=self._runtime)