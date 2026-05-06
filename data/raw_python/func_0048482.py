def get_assessment_part_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment part lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.authoring.AssessmentPartLookupSession)
                - an ``AssessmentPartLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_part_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_lookup()`` is ``true``.*

        """
        if not self.supports_assessment_part_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentPartLookupSession(proxy=proxy, runtime=self._runtime)