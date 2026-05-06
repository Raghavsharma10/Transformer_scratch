def get_objective_admin_session(self):
        """Gets the OsidSession associated with the objective
        administration service.

        return: (osid.learning.ObjectiveAdminSession) - an
                ObjectiveAdminSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_admin() is false
        compliance: optional - This method must be implemented if
                    supports_objective_admin() is true.

        """
        if not self.supports_objective_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveAdminSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session