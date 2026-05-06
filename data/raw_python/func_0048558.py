def get_objective_requisite_session(self):
        """Gets the session for examining objective requisites.

        return: (osid.learning.ObjectiveRequisiteSession) - an
                ObjectiveRequisiteSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_requisite() is false
        compliance: optional - This method must be implemented if
                    supports_objective_requisite() is true.

        """
        if not self.supports_objective_requisite():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveRequisiteSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session