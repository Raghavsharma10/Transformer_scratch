def get_learning_path_session_for_objective_bank(self, objective_bank_id=None):
        """Gets the OsidSession associated with the learning path service
        for the given objective bank.

        arg:    objectiveBankId (osid.id.Id): the Id of the
                ObjectiveBank
        return: (osid.learning.LearningPathSession) - a
                LearningPathSession
        raise:  NotFound - no objective bank found by the given Id
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supporty_learning_path() or
                supports_visible_federation() is false
        compliance: optional - This method must be implemented if
                    supports_learning_path() and
                    supports_visible_federation() are true

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_learning_path():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.LearningPathSession(objective_bank_id, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session