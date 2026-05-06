def get_my_learning_path_session_for_objective_bank(self, objective_bank_id, proxy):
        """Gets the ``OsidSession`` associated with the my learning path service for the given objective bank.

        :param objective_bank_id: the ``Id`` of the ``ObjectiveBank``
        :type objective_bank_id: ``osid.id.Id``
        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: a ``MyLearningPathSession``
        :rtype: ``osid.learning.MyLearningPathSession``
        :raise: ``NotFound`` -- no objective bank found by the given ``Id``
        :raise: ``NullArgument`` -- ``objective_bank_id`` or ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_my_learning_path()`` or ``supports_visible_federation()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_my_learning_path()`` and ``supports_visible_federation()`` are ``true``*

        """
        if not objective_bank_id:
            raise NullArgument
        if not self.supports_my_learning_path():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.MyLearningPathSession(objective_bank_id=objective_bank_id, proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session