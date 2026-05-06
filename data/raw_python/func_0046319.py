def get_bank_hierarchy_design_session(self, proxy):
        """Gets the session designing bank hierarchies.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.BankHierarchyDesignSession) - a
                ``BankHierarchySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bank_hierarchy_design() is
                false``
        *compliance: optional -- This method must be implemented if
        ``supports_bank_hierarchy_design()`` is true.*

        """
        if not self.supports_bank_hierarchy_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BankHierarchyDesignSession(proxy=proxy, runtime=self._runtime)