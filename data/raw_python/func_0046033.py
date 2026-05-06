def get_relationship_admin_session_for_family(self, family_id=None):
        """Gets the ``OsidSession`` associated with the relationship administration service for the given family.

        arg:    family_id (osid.id.Id): the ``Id`` of the ``Family``
        return: (osid.relationship.RelationshipAdminSession) - a
                ``RelationshipAdminSession``
        raise:  NotFound - no family found by the given ``Id``
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_relationship_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_admin()``
            and ``supports_visible_federation()`` are ``true``*

        """
        if not family_id:
            raise NullArgument
        if not self.supports_relationship_admin():
            raise Unimplemented()
        ##
        # Need to include check to see if the familyId is found otherwise raise NotFound
        ##
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.RelationshipAdminSession(family_id,
                                                        proxy=self._proxy,
                                                        runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session