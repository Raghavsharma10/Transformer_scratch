def get_objective_admin_session(self, proxy, *args, **kwargs):
        """Gets the ``OsidSession`` associated with the objective administration service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: an ``ObjectiveAdminSession``
        :rtype: ``osid.learning.ObjectiveAdminSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_objective_admin()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_objective_admin()`` is ``true``.*

        """
        if not self.supports_objective_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ObjectiveAdminSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session