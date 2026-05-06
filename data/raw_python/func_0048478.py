def get_sequence_rule_admin_session(self):
        """Gets the ``OsidSession`` associated with the sequence rule administration service.

        return: (osid.assessment.authoring.SequenceRuleAdminSession) - a
                ``SequenceRuleAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_sequence_rule_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_sequence_rule_admin()`` is ``true``.*

        """
        if not self.supports_sequence_rule_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.SequenceRuleAdminSession(runtime=self._runtime)