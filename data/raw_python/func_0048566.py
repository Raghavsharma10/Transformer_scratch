def get_proficiency_lookup_session_for_objective_bank(self, objective_bank_id=None):
        """Gets the OsidSession associated with the proficiency lookup
        service for the given objective bank.

        arg:    objectiveBankId (osid.id.Id): the Id of the obective
                bank
        return: (osid.learning.ProficiencyLookupSession) - a
                ProficiencyLookupSession
        raise:  NotFound - no ObjectiveBank found by the given Id
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_proficiency_lookup() or
                supports_visible_federation() is false
        compliance: optional - This method must be implemented if
                    supports_proficiency_lookup() and
                    supports_visible_federation() are true

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_proficiency_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ProficiencyLookupSession(objective_bank_id, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session