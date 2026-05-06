def get_relationship_query_session_for_family(self, family_id=None, proxy=None):
        """Gets the ``OsidSession`` associated with the relationship query service for the given family.

        arg:    family_id (osid.id.Id): the ``Id`` of the family
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.RelationshipQuerySession) - a
                ``RelationshipQuerySession``
        raise:  NotFound - no ``Family`` found by the given ``Id``
        raise:  NullArgument - ``family_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_query()``
            and ``supports_visible_federation()`` are ``true``*

        """
        if not family_id:
            raise NullArgument
        if not self.supports_relationship_query():
            raise Unimplemented()
        ##
        # Need to include check to see if the familyId is found otherwise raise NotFound
        ##
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.RelationshipQuerySession(family_id, proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session