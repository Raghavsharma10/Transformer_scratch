def get_activity_lookup_session(self, proxy, *args, **kwargs):
        """Gets the ``OsidSession`` associated with the activity lookup service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ActivityLookupSession``
        :rtype: ``osid.learning.ActivityLookupSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_activity_lookup()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_activity_lookup()`` is ``true``.*

        """
        if not self.supports_activity_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ActivityLookupSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session