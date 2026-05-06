def get_my_learning_path_session(self, proxy):
        """Gets the ``OsidSession`` associated with the my learning path service.

        :param proxy: a proxy
        :type proxy: ``osid.proxy.Proxy``
        :return: a ``MyLearningPathSession``
        :rtype: ``osid.learning.MyLearningPathSession``
        :raise: ``NullArgument`` -- ``proxy`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_my_learning_path()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_my_learning_path()`` is ``true``.*

        """
        if not self.supports_my_learning_path():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.MyLearningPathSession(proxy=proxy, runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session