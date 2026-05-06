def get_objective_hierarchy_session(self, proxy):
        """Gets the session for traversing objective hierarchies.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveHierarchySession) - an
                ``ObjectiveHierarchySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_hierarchy()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_hierarchy()`` is ``true``.*

        """
        if not self.supports_objective_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveHierarchySession(proxy=proxy, runtime=self._runtime)