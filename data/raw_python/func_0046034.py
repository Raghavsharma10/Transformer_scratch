def get_family_lookup_session(self):
        """Gets the ``OsidSession`` associated with the family lookup service.

        return: (osid.relationship.FamilyLookupSession) - a
                ``FamilyLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_family_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_family_lookup()`` is ``true``.*

        """
        if not self.supports_family_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.FamilyLookupSession(proxy=self._proxy,
                                                   runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session