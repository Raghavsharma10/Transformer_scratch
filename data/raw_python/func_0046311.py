def get_assessment_offered_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment offered administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.AssessmentOfferedAdminSession) - an
                ``AssessmentOfferedAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_offered_admin()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_offered_admin()`` is ``true``.*

        """
        if not self.supports_assessment_offered_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentOfferedAdminSession(proxy=proxy, runtime=self._runtime)