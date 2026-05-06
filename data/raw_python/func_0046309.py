def get_assessment_basic_authoring_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment authoring service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentBasicAuthoringSession) - an
                ``AssessmentBasicAuthoringSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_assessment_basic_authoring()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_basic_authoring()`` is ``true``.*

        """
        if not self.supports_assessment_basic_authoring():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentBasicAuthoringSession(proxy=proxy, runtime=self._runtime)