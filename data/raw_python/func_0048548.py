def get_objective_lookup_session(self):
        """Gets the OsidSession associated with the objective lookup
        service.

        return: (osid.learning.ObjectiveLookupSession) - an
                ObjectiveLookupSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_lookup() is false
        compliance: optional - This method must be implemented if
                    supports_objective_lookup() is true.

        """
        if not self.supports_objective_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise  # OperationFailed()
        try:
            session = sessions.ObjectiveLookupSession(runtime=self._runtime)
        except AttributeError:
            raise  # OperationFailed()
        return session