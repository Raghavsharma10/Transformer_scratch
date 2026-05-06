def get_relationship_lookup_session_for_family(self, family_id=None, proxy=None, *args, **kwargs):
        """Gets the ``OsidSession`` associated with the relationship lookup service for the given family.

        arg:    family_id (osid.id.Id): the ``Id`` of the family
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.RelationshipLookupSession) - a
                ``RelationshipLookupSession``
        raise:  NotFound - no ``Family`` found by the given ``Id``
        raise:  NullArgument - ``family_id`` or ``proxy`` is ``null``
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
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.RelationshipLookupSession(family_id, proxy=proxy, runtime=self._runtime, **kwargs)
        except AttributeError:
            raise OperationFailed()
        return session