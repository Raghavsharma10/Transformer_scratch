def get_proficiency_search_session(self, proxy):
        """Gets the ``OsidSession`` associated with the proficiency search service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: a ``ProficiencySearchSession``
        :rtype: ``osid.learning.ProficiencySearchSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_proficiency_search()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_proficiency_search()`` is ``true``.*

        """
        if not self.supports_proficiency_search():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.ProficiencySearchSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session