def get_vault_hierarchy_session(self):
        """Gets the session traversing vault hierarchies.

        return: (osid.authorization.VaultHierarchySession) - a
                ``VaultHierarchySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_vault_hierarchy() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_vault_hierarchy()`` is true.*

        """
        if not self.supports_vault_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.VaultHierarchySession(runtime=self._runtime)