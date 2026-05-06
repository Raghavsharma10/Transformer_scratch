def get_sequence_rule_admin_session_for_bank(self, bank_id):
        """Gets the ``OsidSession`` associated with the sequence rule administration service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        return: (osid.assessment.authoring.SequenceRuleAdminSession) - a
                ``SequenceRuleAdminSession``
        raise:  NotFound - no ``Bank`` found by the given ``Id``
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_sequence_rule_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_sequence_rule_admin()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_sequence_rule_admin():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.SequenceRuleAdminSession(bank_id, runtime=self._runtime)