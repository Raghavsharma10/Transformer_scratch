def get_objective_requisite_assignment_session(self, *args, **kwargs):
        """Gets the session for managing objective requisites.

        return: (osid.learning.ObjectiveRequisiteAssignmentSession) - an
                ObjectiveRequisiteAssignmentSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                supports_objective_requisite_assignment() is false
        compliance: optional - This method must be implemented if
                    supports_objective_requisite_assignment() is true.

        """
        if not self.supports_objective_requisite_assignment():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveRequisiteAssignmentSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session