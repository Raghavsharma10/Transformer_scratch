def get_composition_search_session_for_repository(self, repository_id):
        """Gets a composition search session for the given repository.

        arg:    repository_id (osid.id.Id): the ``Id`` of the repository
        return: (osid.repository.CompositionSearchSession) - a
                ``CompositionSearchSession``
        raise:  NotFound - ``repository_id`` not found
        raise:  NullArgument - ``repository_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_composition_search()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_composition_search()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_composition_search():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.CompositionSearchSession(repository_id, runtime=self._runtime)