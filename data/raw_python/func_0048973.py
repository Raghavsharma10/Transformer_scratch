def get_objective_objective_bank_session(self):
        """Gets the session for retrieving objective to objective bank mappings.

        return: (osid.learning.ObjectiveObjectiveBankSession) - an
                ``ObjectiveObjectiveBankSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_objective_bank()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_objective_bank()`` is ``true``.*

        """
        if not self.supports_objective_objective_bank():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveObjectiveBankSession(runtime=self._runtime)