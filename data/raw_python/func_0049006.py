def get_objective_bank_admin_session(self, proxy):
        """Gets the OsidSession associated with the objective bank administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveBankAdminSession) - an
                ``ObjectiveBankAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_bank_admin() is
                false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_bank_admin()`` is true.*

        """
        if not self.supports_objective_bank_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveBankAdminSession(proxy=proxy, runtime=self._runtime)