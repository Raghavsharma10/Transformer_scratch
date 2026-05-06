def get_sequence_rule_lookup_session(self):
        """Gets the ``OsidSession`` associated with the sequence rule lookup service.

        return: (osid.assessment.authoring.SequenceRuleLookupSession) -
                a ``SequenceRuleLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_sequence_rule_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_sequence_rule_lookup()`` is ``true``.*

        """
        if not self.supports_sequence_rule_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.SequenceRuleLookupSession(runtime=self._runtime)