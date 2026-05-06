def get_objective_objective_bank_assignment_session(self, proxy):
        """Gets the session for assigning objective to objective bank mappings.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ObjectiveObjectiveBankAssignmentSession``
        :rtype: ``osid.learning.ObjectiveObjectiveBankAssignmentSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_objective_objective_bank_assignment()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_objective_objective_bank_assignment()`` is ``true``.*

        """
        if not self.supports_objective_objective_bank_assignment():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ObjectiveObjectiveBankAssignmentSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session