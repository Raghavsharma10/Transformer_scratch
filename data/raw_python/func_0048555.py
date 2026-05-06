def get_objective_hierarchy_session_for_objective_bank(self, objective_bank_id=None, *args, **kwargs):
        """Gets the OsidSession associated with the objective hierarchy
        traversal service for the given objective bank.

        arg:    objectiveBankId (osid.id.Id): the Id of the objective
                bank
        return: (osid.learning.ObjectiveHierarchySession) - an
                ObjectiveHierarchySession
        raise:  NotFound - objectiveBankId not found
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_hierarchy() or
                supports_visible_federation() is false
        compliance: optional - This method must be implemented if
                    supports_objective_hierarchy() and
                    supports_visible_federation() are true.

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_objective_hierarchy():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveHierarchySession(objective_bank_id, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session