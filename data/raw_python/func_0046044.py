def get_relationship_family_assignment_session(self, proxy=None):
        """Gets the ``OsidSession`` associated with assigning relationships to families.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.RelationshipFamilyAssignmentSession)
                - a ``RelationshipFamilyAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_relationship_family_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_relationship_family_assignment()``
            is ``true``.*

        """
        if not self.supports_relationship_family_assignment():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.RelationshipFamilyAssignmentSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session