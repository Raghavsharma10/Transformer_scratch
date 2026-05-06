def get_composition_repository_assignment_session(self, proxy):
        """Gets the session for assigning composition to repository mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.repository.CompositionRepositoryAssignmentSession)
                - a ``CompositionRepositoryAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_composition_repository_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_composition_repository_assignment()`` is ``true``.*

        """
        if not self.supports_composition_repository_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CompositionRepositoryAssignmentSession(proxy=proxy, runtime=self._runtime)