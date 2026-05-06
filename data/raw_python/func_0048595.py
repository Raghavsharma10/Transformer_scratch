def get_proficiency_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the proficiency administration service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: a ``ProficiencyAdminSession``
        :rtype: ``osid.learning.ProficiencyAdminSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_proficiency_admin()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_proficiency_admin()`` is ``true``.*

        """
        if not self.supports_proficiency_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ProficiencyAdminSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session