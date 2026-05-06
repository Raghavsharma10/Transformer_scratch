def get_objective_hierarchy_design_session(self):
        """Gets the session for designing objective hierarchies.

        return: (osid.learning.ObjectiveHierarchyDesignSession) - an
                ObjectiveHierarchyDesignSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_hierarchy_design() is
                false
        compliance: optional - This method must be implemented if
                    supports_objective_hierarchy_design() is true.

        """
        if not self.supports_objective_hierarchy_design():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveHierarchyDesignSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session