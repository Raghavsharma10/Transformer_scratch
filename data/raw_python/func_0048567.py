def get_proficiency_admin_session_for_objective_bank(self, objective_bank_id=None):
        """Gets the OsidSession associated with the proficiency
        administration service for the given objective bank.

        arg:    objectiveBankId (osid.id.Id): the Id of the
                ObjectiveBank
        return: (osid.learning.ProficiencyAdminSession) - a
                ProficiencyAdminSession
        raise:  NotFound - no objective bank found by the given Id
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_proficiency_admin() or
                supports_visible_federation() is false
        compliance: optional - This method must be implemented if
                    supports_proficiency_admin() and
                    supports_visible_federation() are true

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_proficiency_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ProficiencyAdminSession(objective_bank_id, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session