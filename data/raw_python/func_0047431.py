def get_proxy_session(self):
        """Gets a ``ProxySession`` which is responsible for acquiring authentication credentials on behalf of a service client.

        :return: a proxy session for this service
        :rtype: ``osid.proxy.ProxySession``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``Unimplemented`` -- ``supports_proxy()`` is ``false``

        *compliance: optional -- This method must be implemented if ``supports_proxy()`` is ``true``.*

        """
        if not self.supports_proxy():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise  # OperationFailed()
        try:
            session = sessions.ProxySession()
        except AttributeError:
            raise  # OperationFailed()
        return session