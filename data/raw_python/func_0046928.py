def get_composition_search_session(self):
        """Gets a composition search session.

        return: (osid.repository.CompositionSearchSession) - a
                ``CompositionSearchSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_composition_search()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_composition_search()`` is ``true``.*

        """
        if not self.supports_composition_search():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CompositionSearchSession(runtime=self._runtime)