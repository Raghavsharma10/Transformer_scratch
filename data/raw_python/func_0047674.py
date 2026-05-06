def get_logging_session_for_log(self, log_id, proxy):
        """Gets the ``OsidSession`` associated with the logging service for the given log.

        arg:    log_id (osid.id.Id): the ``Id`` of the ``Log``
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LoggingSession) - a ``LoggingSession``
        raise:  NotFound - no ``Log`` found by the given ``Id``
        raise:  NullArgument - ``log_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_logging()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_logging()`` and ``supports_visible_federation()`` are
        ``true``*

        """
        if not self.supports_logging():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.LoggingSession(log_id, proxy, self._runtime)