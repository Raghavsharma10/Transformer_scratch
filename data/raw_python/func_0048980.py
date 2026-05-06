def get_objective_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the objective lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveLookupSession) - an
                ``ObjectiveLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_lookup()`` is ``true``.*

        """
        if not self.supports_objective_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveLookupSession(proxy=proxy, runtime=self._runtime)