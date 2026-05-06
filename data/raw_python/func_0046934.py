def get_repository_hierarchy_session(self):
        """Gets the repository hierarchy traversal session.

        return: (osid.repository.RepositoryHierarchySession) - ``a
                RepositoryHierarchySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_repository_hierarchy()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_repository_hierarchy()`` is ``true``.*

        """
        if not self.supports_repository_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.RepositoryHierarchySession(runtime=self._runtime)