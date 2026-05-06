def get_repository_query_session(self):
        """Gets the repository query session.

        return: (osid.repository.RepositoryQuerySession) - a
                ``RepositoryQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_repository_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_repository_query()`` is ``true``.*

        """
        if not self.supports_repository_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.RepositoryQuerySession(runtime=self._runtime)