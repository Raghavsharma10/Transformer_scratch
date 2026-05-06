def get_objective_objective_bank_assignment_session(self):
        """Gets the session for assigning objective to objective bank mappings.

        return: (osid.learning.ObjectiveObjectiveBankAssignmentSession)
                - an ``ObjectiveObjectiveBankAssignmentSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_objective_objective_bank_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_objective_bank_assignment()`` is ``true``.*

        """
        if not self.supports_objective_objective_bank_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveObjectiveBankAssignmentSession(runtime=self._runtime)