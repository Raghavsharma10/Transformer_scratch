def get_objective_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the objective query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveQuerySession) - an
                ``ObjectiveQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_query()`` is ``true``.*

        """
        if not self.supports_objective_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveQuerySession(proxy=proxy, runtime=self._runtime)