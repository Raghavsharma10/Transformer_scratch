def get_vault_hierarchy_design_session(self):
        """Gets the session designing vault hierarchies.

        return: (osid.authorization.VaultHierarchyDesignSession) - a
                ``VaultHierarchyDesignSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_vault_hierarchy_design() is
                false``
        *compliance: optional -- This method must be implemented if
        ``supports_vault_hierarchy_design()`` is true.*

        """
        if not self.supports_vault_hierarchy_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.VaultHierarchyDesignSession(runtime=self._runtime)