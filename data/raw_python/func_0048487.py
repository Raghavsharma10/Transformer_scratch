def get_assessment_part_admin_session_for_bank(self, bank_id, proxy):
        """Gets the ``OsidSession`` associated with the assessment part administration service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.authoring.AssessmentPartAdminSession) -
                an ``AssessmentPartAdminSession``
        raise:  NotFound - no ``Bank`` found by the given ``Id``
        raise:  NullArgument - ``bank_id or proxy is null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_part_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_admin()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_assessment_part_admin():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssessmentPartAdminSession(bank_id, proxy, self._runtime)