def get_proficiency_objective_bank_session(self, proxy):
        """Gets the ``OsidSession`` to lookup proficiency/objective bank mappings.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: a ``ProficiencyObjectiveBankSession``
        :rtype: ``osid.learning.ProficiencyObjectiveBankSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_proficiency_objective_bank()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_proficiency_objective_bank()`` is ``true``.*

        """
        if not self.supports_proficiency_objective_bank():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ProficiencyObjectiveBankSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session