def get_proficiency_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the proficiency query service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: a ``ProficiencyQuerySession``
        :rtype: ``osid.learning.ProficiencyQuerySession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_proficiency_query()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_proficiency_query()`` is ``true``.*

        """
        if not self.supports_proficiency_query():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ProficiencyQuerySession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session