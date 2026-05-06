def get_authorization_admin_session(self):
        """Gets the ``OsidSession`` associated with the authorization administration service.

        return: (osid.authorization.AuthorizationAdminSession) - an
                ``AuthorizationAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_authorization_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization_admin()`` is ``true``.*

        """
        if not self.supports_authorization_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AuthorizationAdminSession(runtime=self._runtime)