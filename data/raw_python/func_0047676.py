def get_log_entry_lookup_session_for_log(self, log_id, proxy):
        """Gets the ``OsidSession`` associated with the log reading service for the given log.

        arg:    log_id (osid.id.Id): the ``Id`` of the ``Log``
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.logging.LogEntryLookupSession) - a
                ``LogEntryLookupSession``
        raise:  NotFound - no ``Log`` found by the given ``Id``
        raise:  NullArgument - ``log_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_entry_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_entry_lookup()`` and
        ``supports_visible_federation()`` are ``true``*

        """
        if not self.supports_log_entry_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.LogEntryLookupSession(log_id, proxy, self._runtime)