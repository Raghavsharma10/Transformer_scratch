def get_objective_search_session(self):
        """Gets the OsidSession associated with the objective search
        service.

        return: (osid.learning.ObjectiveSearchSession) - an
                ObjectiveSearchSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_search() is false
        compliance: optional - This method must be implemented if
                    supports_objective_search() is true.

        """
        if not self.supports_objective_search():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveSearchSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session