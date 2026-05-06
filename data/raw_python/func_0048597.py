def get_proficiency_objective_bank_assignment_session(self, proxy):
        """Gets the ``OsidSession`` associated with assigning proficiencies to objective banks.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: a ``ProficiencyObjectiveBankAssignmentSession``
        :rtype: ``osid.learning.ProficiencyObjectiveBankAssignmentSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_proficiency_objective_bank_assignment()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_proficiency_objective_bank_assignment()`` is ``true``.*

        """
        if not self.supports_proficiency_objective_bank_assignment():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ProficiencyObjectiveBankAssignmentSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session