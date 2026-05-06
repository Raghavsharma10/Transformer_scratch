def get_relationship_lookup_session(self):
        """Gets the ``OsidSession`` associated with the relationship lookup service.

        return: (osid.relationship.RelationshipLookupSession) - a
                ``RelationshipLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_lookup()``
            is ``true``.*

        """
        if not self.supports_relationship_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.RelationshipLookupSession(proxy=self._proxy,
                                                         runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session