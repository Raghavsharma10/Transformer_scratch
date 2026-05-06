def get_assessment_part_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment part administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.authoring.AssessmentPartAdminSession) -
                an ``AssessmentPartAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_part_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_admin()`` is ``true``.*

        """
        if not self.supports_assessment_part_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentPartAdminSession(proxy=proxy, runtime=self._runtime)