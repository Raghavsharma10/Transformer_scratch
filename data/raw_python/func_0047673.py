def get_logging_session(self, proxy):
        """Gets the ``OsidSession`` associated with the logging service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LoggingSession) - a ``LoggingSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_logging()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_logging()`` is ``true``.*

        """
        if not self.supports_logging():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.LoggingSession(proxy=proxy, runtime=self._runtime)