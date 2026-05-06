def get_repository_hierarchy_design_session(self):
        """Gets the repository hierarchy design session.

        return: (osid.repository.RepositoryHierarchyDesignSession) - a
                ``RepostoryHierarchyDesignSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_repository_hierarchy_design()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_repository_hierarchy_design()`` is ``true``.*

        """
        if not self.supports_repository_hierarchy_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.RepositoryHierarchyDesignSession(runtime=self._runtime)