def get_authorization_session(self):
        """Gets an ``AuthorizationSession`` which is responsible for performing authorization checks.

        return: (osid.authorization.AuthorizationSession) - an
                authorization session for this service
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_authorization()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization()`` is ``true``.*

        """
        if not self.supports_authorization():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AuthorizationSession(runtime=self._runtime)