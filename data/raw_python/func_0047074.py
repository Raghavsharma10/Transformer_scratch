def get_authorization_lookup_session(self):
        """Gets the ``OsidSession`` associated with the authorization lookup service.

        return: (osid.authorization.AuthorizationLookupSession) - an
                ``AuthorizationLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_authorization_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization_lookup()`` is ``true``.*

        """
        if not self.supports_authorization_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AuthorizationLookupSession(runtime=self._runtime)