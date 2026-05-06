def get_objective_bank_hierarchy_design_session(self):
        """Gets the session designing objective bank hierarchies.

        return: (osid.learning.ObjectiveBankHierarchyDesignSession) - an
                ObjectiveBankHierarchyDesignSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                supports_objective_bank_hierarchy_design() is false
        compliance: optional - This method must be implemented if
                    supports_objective_bank_hierarchy_design() is true.

        """
        if not self.supports_objective_bank_hierarchy_design():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveBankHierarchyDesignSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session