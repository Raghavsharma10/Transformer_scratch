def get_objective_admin_session_for_objective_bank(self, objective_bank_id, proxy, *args, **kwargs):
        """Gets the ``OsidSession`` associated with the objective admin service for the given objective bank.

        :param objective_bank_id: the ``Id`` of the objective bank
        :type objective_bank_id: ``osid.id.Id``
        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: ``an _objective_admin_session``
        :rtype: ``osid.learning.ObjectiveAdminSession``
        :raise: ``NotFound`` -- ``objective_bank_id`` not found
        :raise: ``NullArgument`` -- ``objective_bank_id`` or ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- ``unable to complete request``
        :raise: ``Unimplemented`` -- ``supports_objective_admin()`` or ``supports_visible_federation()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_objective_admin()`` and ``supports_visible_federation()`` are ``true``.*

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_objective_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ObjectiveAdminSession(objective_bank_id=objective_bank_id, proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session