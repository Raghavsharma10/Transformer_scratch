def get_composition_query_session(self):
        """Gets a composition query session.

        return: (osid.repository.CompositionQuerySession) - a
                ``CompositionQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_composition_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_composition_query()`` is ``true``.*

        """
        if not self.supports_composition_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CompositionQuerySession(runtime=self._runtime)