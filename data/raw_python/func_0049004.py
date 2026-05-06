def get_proficiency_objective_bank_assignment_session(self, proxy):
        """Gets the ``OsidSession`` associated with assigning proficiencies to objective banks.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return:
                (osid.learning.ProficiencyObjectiveBankAssignmentSession
                ) - a ``ProficiencyObjectiveBankAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_proficiency_objective_bank_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_proficiency_objective_bank_assignment()`` is
        ``true``.*

        """
        if not self.supports_proficiency_objective_bank_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ProficiencyObjectiveBankAssignmentSession(proxy=proxy, runtime=self._runtime)