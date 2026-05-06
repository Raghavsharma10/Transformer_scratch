def get_activity_query_session_for_objective_bank(self, objective_bank_id, proxy):
        """Gets the ``OsidSession`` associated with the activity query service for the given objective bank.

        :param objective_bank_id: the ``Id`` of the objective bank
        :type objective_bank_id: ``osid.id.Id``
        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ActivityQuerySession``
        :rtype: ``osid.learning.ActivityQuerySession``
        :raise: ``NotFound`` -- ``objective_bank_id`` not found
        :raise: ``NullArgument`` -- ``objective_bank_id`` or ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- ``unable to complete request``
        :raise: ``Unimplemented`` -- ``supports_activity_query()`` or ``supports_visible_federation()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_activity_query()`` and ``supports_visible_federation()`` are ``true``.*

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_activity_query():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ActivityQuerySession(objective_bank_id=objective_bank_id, proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session