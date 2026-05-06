def get_proxy_session(self):
        """Gets a ``ProxySession`` which is responsible for acquiring authentication credentials on behalf of a service client.

        return: (osid.proxy.ProxySession) - a proxy session for this
                service
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_proxy()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_proxy()`` is ``true``.*

        """
        if not self.supports_proxy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ProxySession(runtime=self._runtime)