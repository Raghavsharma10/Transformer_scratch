def get_assessment_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentQuerySession) - an
                ``AssessmentQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_query()`` is ``true``.*

        """
        if not self.supports_assessment_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentQuerySession(proxy=proxy, runtime=self._runtime)