def get_objective_bank_admin_session(self, proxy, *args, **kwargs):
        """Gets the OsidSession associated with the objective bank administration service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ObjectiveBankAdminSession``
        :rtype: ``osid.learning.ObjectiveBankAdminSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_objective_bank_admin() is false``

        *compliance: optional -- This method must be implemented if ``supports_objective_bank_admin()`` is true.*

        """
        if not self.supports_objective_bank_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ObjectiveBankAdminSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session