def get_objective_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the objective query service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ObjectiveQuerySession``
        :rtype: ``osid.learning.ObjectiveQuerySession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_objective_query()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_objective_query()`` is ``true``.*

        """
        if not self.supports_objective_query():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ObjectiveQuerySession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session