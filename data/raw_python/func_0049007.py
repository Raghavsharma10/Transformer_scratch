def get_objective_bank_hierarchy_session(self, proxy):
        """Gets the session traversing objective bank hierarchies.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveBankHierarchySession) - an
                ``ObjectiveBankHierarchySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_bank_hierarchy() is
                false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_bank_hierarchy()`` is true.*

        """
        if not self.supports_objective_bank_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveBankHierarchySession(proxy=proxy, runtime=self._runtime)