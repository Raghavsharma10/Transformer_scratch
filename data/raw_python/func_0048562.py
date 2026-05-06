def get_activity_search_session_for_objective_bank(self, objective_bank_id=None):
        """Gets the OsidSession associated with the activity search service
        for the given objective bank.

        arg:    objectiveBankId (osid.id.Id): the Id of the objective
                bank
        return: (osid.learning.ActivitySearchSession) - an
                ActivitySearchSession
        raise:  NotFound - objectiveBankId not found
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_activity_search() or
                supports_visible_federation() is false
        compliance: optional - This method must be implemented if
                    supports_activity_search() and
                    supports_visible_federation() are true.

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_activity_search():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ActivitySearchSession(objective_bank_id, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session