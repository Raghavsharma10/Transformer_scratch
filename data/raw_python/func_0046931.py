def get_repository_lookup_session(self):
        """Gets the repository lookup session.

        return: (osid.repository.RepositoryLookupSession) - a
                ``RepositoryLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_repository_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_repository_lookup()`` is ``true``.*

        """
        if not self.supports_repository_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.RepositoryLookupSession(runtime=self._runtime)