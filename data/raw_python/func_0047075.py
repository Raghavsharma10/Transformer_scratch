def get_authorization_query_session(self):
        """Gets the ``OsidSession`` associated with the authorization query service.

        return: (osid.authorization.AuthorizationQuerySession) - an
                ``AuthorizationQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_authorization_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_authorization_query()`` is ``true``.*

        """
        if not self.supports_authorization_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AuthorizationQuerySession(runtime=self._runtime)