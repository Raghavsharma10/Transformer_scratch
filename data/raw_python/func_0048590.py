def get_activity_objective_bank_assignment_session(self, proxy):
        """Gets the session for assigning activity to objective bank mappings.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ActivityObjectiveBankAssignmentSession``
        :rtype: ``osid.learning.ActivityObjectiveBankAssignmentSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_activity_objective_bank_assignment()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_activity_objective_bank_assignment()`` is ``true``.*

        """
        if not self.supports_activity_objective_bank_assignment():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ActivityObjectiveBankAssignmentSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session