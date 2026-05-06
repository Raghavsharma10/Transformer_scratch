def get_repository_admin_session(self):
        """Gets the repository administrative session for creating, updating and deleteing repositories.

        return: (osid.repository.RepositoryAdminSession) - a
                ``RepositoryAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_repository_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_repository_admin()`` is ``true``.*

        """
        if not self.supports_repository_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.RepositoryAdminSession(runtime=self._runtime)