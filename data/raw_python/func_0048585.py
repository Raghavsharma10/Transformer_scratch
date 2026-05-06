def get_activity_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the activity query service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ActivityQuerySession``
        :rtype: ``osid.learning.ActivityQuerySession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_activity_query()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_activity_query()`` is ``true``.*

        """
        if not self.supports_activity_query():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ActivityQuerySession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session