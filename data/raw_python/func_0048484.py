def get_assessment_part_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment part query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.authoring.AssessmentPartQuerySession) -
                an ``AssessmentPartQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_part_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_query()`` is ``true``.*

        """
        if not self.supports_assessment_part_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentPartQuerySession(proxy=proxy, runtime=self._runtime)