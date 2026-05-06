def get_objective_objective_bank_session(self, proxy):
        """Gets the session for retrieving objective to objective bank mappings.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ObjectiveObjectiveBankSession``
        :rtype: ``osid.learning.ObjectiveObjectiveBankSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_objective_objective_bank()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_objective_objective_bank()`` is ``true``.*

        """
        if not self.supports_objective_objective_bank():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ObjectiveObjectiveBankSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session