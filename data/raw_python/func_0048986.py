def get_objective_hierarchy_design_session(self, proxy):
        """Gets the session for designing objective hierarchies.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveHierarchyDesignSession) - an
                ``ObjectiveHierarchyDesignSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_objective_hierarchy_design()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_hierarchy_design()`` is ``true``.*

        """
        if not self.supports_objective_hierarchy_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveHierarchyDesignSession(proxy=proxy, runtime=self._runtime)