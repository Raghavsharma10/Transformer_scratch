def get_objective_lookup_session_for_objective_bank(self, objective_bank_id=None):
        """Gets the OsidSession associated with the objective lookup
        service for the given objective bank.

        arg:    objectiveBankId (osid.id.Id): the Id of the objective
                bank
        return: (osid.learning.ObjectiveLookupSession) - an
                ObjectiveLookupSession
        raise:  NotFound - objectiveBankId not found
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_lookup() or
                supports_visible_federation() is false
        compliance: optional - This method must be implemented if
                    supports_objective_lookup() and
                    supports_visible_federation() are true.

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_objective_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveLookupSession(objective_bank_id, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session