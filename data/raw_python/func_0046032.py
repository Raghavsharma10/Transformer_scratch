def get_relationship_lookup_session_for_family(self, family_id=None):
        """Gets the ``OsidSession`` associated with the relationship lookup service for the given family.

        arg:    family_id (osid.id.Id): the ``Id`` of the family
        return: (osid.relationship.RelationshipLookupSession) - a
                ``RelationshipLookupSession``
        raise:  NotFound - no ``Family`` found by the given ``Id``
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_lookup()``
            and ``supports_visible_federation()`` are ``true``*

        """
        if not family_id:
            raise NullArgument
        if not self.supports_relationship_lookup():
            raise Unimplemented()
        ##
        # Need to include check to see if the familyId is found otherwise raise NotFound
        ##
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.RelationshipLookupSession(family_id,
                                                         proxy=self._proxy,
                                                         runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session