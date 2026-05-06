def get_objective_bank_search_session(self, proxy):
        """Gets the OsidSession associated with the objective bank search service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ObjectiveBankSearchSession``
        :rtype: ``osid.learning.ObjectiveBankSearchSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_objective_bank_search() is false``

        *compliance: optional -- This method must be implemented if ``supports_objective_bank_search()`` is true.*

        """
        if not self.supports_objective_bank_search():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ObjectiveBankSearchSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session