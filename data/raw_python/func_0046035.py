def get_family_admin_session(self):
        """Gets the ``OsidSession`` associated with the family administrative service.

        return: (osid.relationship.FamilyAdminSession) - a
                ``FamilyAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_family_admin()`` is ``false``
        *compliance: optional -- This method must be implemented if ``supports_family_admin()`` is ``true``.*

        """
        if not self.supports_family_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.FamilyAdminSession(proxy=self._proxy,
                                                  runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session